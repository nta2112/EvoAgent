from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    CLIPProcessor,
    CLIPModel,
    WhisperProcessor,
    WhisperForConditionalGeneration
)

import torch

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IGNORE_INDEX
)
from llava.conversation import conv_templates, SeparatorStyle


def _force_remove_all_hooks(module):
    """
    Xóa TOÀN BỘ accelerate hooks khỏi module và tất cả sub-modules.
    Khôi phục forward method gốc và xóa triệt để _hf_hook, _old_forward.
    """
    if module is None:
        return

    # 1. Thử dùng hàm chuẩn của accelerate nếu có
    try:
        from accelerate.hooks import remove_hook_from_module
        remove_hook_from_module(module, recurse=True)
    except Exception:
        pass

    # 2. Quét thủ công từng submodule để đảm bảo sạch 100%
    for mod in module.modules():
        # Khôi phục forward gốc từ _old_forward (nơi accelerate lưu forward ban đầu)
        if hasattr(mod, "_old_forward"):
            try:
                mod.forward = mod._old_forward
            except Exception:
                pass
            try:
                delattr(mod, "_old_forward")
            except Exception:
                pass
        elif "forward" in mod.__dict__:
            # Nếu forward bị gán ở instance level (closure new_forward) mà không có _old_forward,
            # xóa nó khỏi instance __dict__ để tự động quay về class-level forward method
            try:
                del mod.__dict__["forward"]
            except Exception:
                pass

        # Gỡ hook sạch sẽ
        if hasattr(mod, "_hf_hook"):
            hook = mod._hf_hook
            try:
                if hasattr(hook, "detach_hook"):
                    hook.detach_hook(mod)
            except Exception:
                pass
            try:
                delattr(mod, "_hf_hook")
            except Exception:
                pass

        # Xóa các thuộc tính khác do accelerate thêm vào instance dict
        for attr in ["to", "cuda", "npu", "xpu", "mlu", "sdaa", "musa"]:
            if attr in mod.__dict__:
                try:
                    del mod.__dict__[attr]
                except Exception:
                    pass


def load_default(path_dict):
    # Required keys in path_dict
    required_keys = [
        "clip_model_path", "whisper_model_path", "llava_model_path",
        "clip_cache_dir", "whisper_cache_dir", "llava_cache_dir"
    ]
    for key in required_keys:
        if key not in path_dict:
            raise ValueError(f"Missing required key in path_dict: {key}")

    # Load CLIP
    clip_model = CLIPModel.from_pretrained(
        path_dict["clip_model_path"],
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=path_dict["clip_cache_dir"]
    )
    clip_processor = CLIPProcessor.from_pretrained(
        path_dict["clip_model_path"],
        cache_dir=path_dict["clip_cache_dir"]
    )

    # Load Whisper
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        path_dict["whisper_model_path"],
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=path_dict["whisper_cache_dir"]
    )
    whisper_processor = WhisperProcessor.from_pretrained(
        path_dict["whisper_model_path"],
        cache_dir=path_dict["whisper_cache_dir"]
    )

    # Load LLaVA
    overwrite_config = path_dict.get("overwrite_config", {})
    llava_model_name = get_model_name_from_path(path_dict["llava_model_path"]) or "llava_qwen"
    torch_dtype = path_dict.get("torch_dtype", torch.float16)
    attn_impl = path_dict.get("attn_implementation", "sdpa")
    llava_device_map = path_dict.get("llava_device_map", "auto")
    max_memory = path_dict.get("max_memory", None)

    # -------------------------------------------------------------------
    # Bảo vệ vocab_size: Ngăn builder.py tự ý thu nhỏ embedding layer từ
    # 152064 xuống 151647 (do len(tokenizer) == 151647), gây lệch shape với
    # weights_map [152064, 3584] của Accelerate khi offload/dispatch.
    # -------------------------------------------------------------------
    from transformers import PreTrainedModel
    orig_resize_fn = PreTrainedModel.resize_token_embeddings

    def safe_resize_token_embeddings(self, new_num_tokens=None, pad_to_multiple_of=None):
        if new_num_tokens is not None:
            model_vocab_size = getattr(self.config, "vocab_size", 0)
            if model_vocab_size > 0 and new_num_tokens <= model_vocab_size:
                print(f"[Embedding] Giữ nguyên vocab_size={model_vocab_size}, bỏ qua thu nhỏ xuống {new_num_tokens}")
                return self.get_input_embeddings()
        return orig_resize_fn(self, new_num_tokens=new_num_tokens, pad_to_multiple_of=pad_to_multiple_of)

    PreTrainedModel.resize_token_embeddings = safe_resize_token_embeddings

    try:
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            model_path=path_dict["llava_model_path"],
            model_base=None,
            model_name=llava_model_name,
            device_map=llava_device_map,
            torch_dtype="float16",
            attn_implementation=attn_impl,
            overwrite_config=overwrite_config,
            max_memory=max_memory
        )
    finally:
        PreTrainedModel.resize_token_embeddings = orig_resize_fn

    # Đảm bảo lm_head có đúng shape theo config nếu lỡ bị thu nhỏ ở bất cứ đâu
    expected_vocab = getattr(model.config, "vocab_size", None)
    if expected_vocab is not None and hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        if model.lm_head.weight.shape[0] != expected_vocab:
            print(f"⚠️ lm_head shape {model.lm_head.weight.shape} lệch với vocab_size {expected_vocab}, khôi phục...")
            orig_resize_fn(model, expected_vocab)

    # -------------------------------------------------------------------
    # Fix vision tower: đảm bảo không còn meta tensor
    # -------------------------------------------------------------------
    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        # Kiểm tra xem vision tower có tham số meta không
        has_meta = any(p.is_meta for p in vision_tower.parameters())

        if has_meta or not getattr(vision_tower, "is_loaded", False):
            print(f"[VT] Vision tower chưa load hoặc còn meta tensor. Đang reload lên cuda:0...")

            # Reset is_loaded để tránh load_model() early return nếu has_meta == True
            vision_tower.is_loaded = False

            # Phương án 1: Dùng LLaVA's own load_model() với device tường minh
            loaded_ok = False
            try:
                vision_tower.load_model(device_map={"": "cuda:0"})
                vision_tower.to(dtype=torch.float16)
                if not any(p.is_meta for p in vision_tower.parameters()):
                    loaded_ok = True
                    print("✅ Vision tower loaded via LLaVA load_model() on cuda:0")
            except Exception as e1:
                print(f"⚠️ load_model() thất bại: {e1}")

            # Phương án 2: Tải SiglipVisionModel từ LLaVA hoặc SiglipVisionModel từ transformers
            if not loaded_ok:
                try:
                    from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionModel
                    vt_name = getattr(
                        vision_tower, "vision_tower_name",
                        "google/siglip-so400m-patch14-384"
                    )
                    vt_model = SigLipVisionModel.from_pretrained(
                        vt_name,
                        torch_dtype=torch_dtype,
                        device_map={"": "cuda:0"}
                    )
                    vt_model.eval()
                    vt_model.requires_grad_(False)
                    vision_tower.vision_tower = vt_model
                    vision_tower.is_loaded = True
                    loaded_ok = True
                    print(f"✅ Vision tower reloaded via LLaVA SigLipVisionModel on cuda:0 ({vt_name})")
                except Exception as e2a:
                    print(f"⚠️ LLaVA SigLipVisionModel fallback thất bại: {e2a}, thử transformers...")
                    try:
                        from transformers import SiglipVisionModel
                        vt_name = getattr(
                            vision_tower, "vision_tower_name",
                            "google/siglip-so400m-patch14-384"
                        )
                        vt_model = SiglipVisionModel.from_pretrained(
                            vt_name,
                            torch_dtype=torch_dtype,
                            device_map={"": "cuda:0"}
                        )
                        vt_model.eval()
                        vt_model.requires_grad_(False)
                        vision_tower.vision_tower = vt_model
                        vision_tower.is_loaded = True
                        loaded_ok = True
                        print(f"✅ Vision tower reloaded via transformers SiglipVisionModel on cuda:0 ({vt_name})")
                    except Exception as e2b:
                        print(f"⚠️ transformers SiglipVisionModel fallback thất bại: {e2b}")

            # Phương án 3: AutoModel (generic fallback)
            if not loaded_ok:
                try:
                    from transformers import AutoModel
                    vt_name = getattr(
                        vision_tower, "vision_tower_name",
                        "google/siglip-so400m-patch14-384"
                    )
                    vt_model = AutoModel.from_pretrained(
                        vt_name,
                        torch_dtype=torch_dtype,
                        device_map={"": "cuda:0"}
                    )
                    vt_model.eval()
                    vt_model.requires_grad_(False)
                    vision_tower.vision_tower = vt_model
                    vision_tower.is_loaded = True
                    print(f"✅ Vision tower reloaded via AutoModel on cuda:0 ({vt_name})")
                except Exception as e3:
                    print(f"⚠️ AutoModel fallback cũng thất bại: {e3}.")
        else:
            print("✅ Vision tower đã được load sẵn, không có meta tensor.")

        # Xóa toàn bộ accelerate hooks khỏi vision_tower và inner model
        _force_remove_all_hooks(vision_tower)
        inner_vt = getattr(vision_tower, "vision_tower", None)
        if inner_vt is not None and isinstance(inner_vt, torch.nn.Module):
            _force_remove_all_hooks(inner_vt)
        print("✅ Đã xóa tất cả accelerate hooks khỏi vision_tower và inner model")

    # Xóa hooks khỏi mm_projector
    mm_projector = getattr(model.get_model(), "mm_projector", None)
    if mm_projector is not None:
        _force_remove_all_hooks(mm_projector)
        print("✅ Đã xóa tất cả accelerate hooks khỏi mm_projector")

    # -------------------------------------------------------------------
    # Patch encode_images: an toàn, tránh hook và xử lý device chính xác
    # -------------------------------------------------------------------
    def safe_encode_images(images):
        # Bắt lỗi nếu images là meta tensor (do bug upstream)
        if images.is_meta:
            raise RuntimeError(
                "BUG UPSTREAM: images là meta tensor khi vào safe_encode_images. "
                "Kiểm tra load_and_sample_video trong ReAgentV.py — "
                "dev không được là meta device."
            )

        vt = model.get_model().get_vision_tower()
        inner_vt = getattr(vt, "vision_tower", vt)

        # Lấy device từ inner model thật sự (không phải wrapper có thể còn meta)
        try:
            vt_device = next(
                p.device for p in inner_vt.parameters() if not p.is_meta
            )
        except StopIteration:
            vt_device = torch.device("cuda:0")

        images = images.to(device=vt_device, dtype=torch.float16)

        with torch.no_grad():
            try:
                image_features = vt(images)
            except Exception:
                image_features = type(vt).forward(vt, images)

        proj = model.get_model().mm_projector
        try:
            proj_device = next(
                p.device for p in proj.parameters() if not p.is_meta
            )
        except StopIteration:
            proj_device = vt_device

        image_features = image_features.to(device=proj_device, dtype=torch.float16)

        with torch.no_grad():
            try:
                image_features = proj(image_features)
            except Exception:
                image_features = type(proj).forward(proj, image_features)

        return image_features

    model.encode_images = safe_encode_images

    model.eval()

    # Chat template
    conv_template = "qwen_1_5"

    return {
        "clip_model": clip_model,
        "clip_processor": clip_processor,
        "whisper_model": whisper_model,
        "whisper_processor": whisper_processor,
        "tokenizer": tokenizer,
        "model": model,
        "image_processor": image_processor,
        "max_length": max_length,
        "conv_template": conv_template
    }
