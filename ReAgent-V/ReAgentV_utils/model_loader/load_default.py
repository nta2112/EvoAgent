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
    Xóa TOÀN BỘ accelerate hooks (_hf_hook) khỏi module và tất cả sub-modules.
    Thao tác trực tiếp trên thuộc tính để tránh lỗi với meta tensor.
    """
    for mod in module.modules():
        if hasattr(mod, "_hf_hook"):
            hf_hook = mod._hf_hook
            # Khôi phục forward method gốc nếu có
            old_fwd = getattr(hf_hook, "old_forward", None)
            if old_fwd is not None and callable(old_fwd):
                mod.forward = old_fwd
            # Xóa hook attribute trực tiếp
            try:
                del mod._hf_hook
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

    # -------------------------------------------------------------------
    # Fix vision tower: đảm bảo không còn meta tensor
    # -------------------------------------------------------------------
    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        # Kiểm tra xem vision tower có tham số meta không
        has_meta = any(p.is_meta for p in vision_tower.parameters())

        if has_meta or not getattr(vision_tower, "is_loaded", False):
            print(f"[VT] Vision tower chưa load hoặc còn meta tensor. Đang reload lên cuda:0...")

            # Phương án 1: Dùng LLaVA's own load_model() với device tường minh
            loaded_ok = False
            try:
                vision_tower.load_model(device_map={"": "cuda:0"})
                vision_tower.to(dtype=torch.float16)
                loaded_ok = True
                print("✅ Vision tower loaded via LLaVA load_model() on cuda:0")
            except Exception as e1:
                print(f"⚠️ load_model() thất bại: {e1}")

            # Phương án 2: Tải SiglipVisionModel trực tiếp (tên đúng: SiglipVisionModel)
            if not loaded_ok:
                try:
                    # QUAN TRỌNG: tên đúng là SiglipVisionModel (chữ 'g' thường),
                    # KHÔNG PHẢI SigLipVisionModel (chữ 'L' hoa) - lỗi trước đây!
                    from transformers import SiglipVisionModel
                    vt_name = getattr(
                        vision_tower, "vision_tower_name",
                        "google/siglip-so400m-patch14-384"
                    )
                    vt_model = SiglipVisionModel.from_pretrained(
                        vt_name,
                        torch_dtype=torch_dtype,
                        device_map={"": "cuda:0"}   # Explicit device, NOT "auto"
                    )
                    vt_model.eval()
                    vt_model.requires_grad_(False)
                    vision_tower.vision_tower = vt_model
                    vision_tower.is_loaded = True
                    loaded_ok = True
                    print(f"✅ Vision tower reloaded via SiglipVisionModel on cuda:0 ({vt_name})")
                except Exception as e2:
                    print(f"⚠️ SiglipVisionModel fallback thất bại: {e2}")

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
                    print(f"⚠️ AutoModel fallback cũng thất bại: {e3}. "
                          "Inference có thể bị lỗi meta tensor.")
        else:
            print("✅ Vision tower đã được load sẵn, không có meta tensor.")

        # Xóa toàn bộ accelerate hooks khỏi vision_tower
        _force_remove_all_hooks(vision_tower)
        print("✅ Đã xóa tất cả accelerate hooks khỏi vision_tower")

    # Xóa hooks khỏi mm_projector
    mm_projector = getattr(model.get_model(), "mm_projector", None)
    if mm_projector is not None:
        _force_remove_all_hooks(mm_projector)
        print("✅ Đã xóa tất cả accelerate hooks khỏi mm_projector")

    # -------------------------------------------------------------------
    # Patch encode_images: bypass hook bằng class-level forward call
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

        # Gọi class-level forward để bypass hook trên vt.__call__
        # type(vt).forward(vt, images) = LLaVA wrapper forward, không qua _hf_hook
        with torch.no_grad():
            image_features = type(vt).forward(vt, images)

        proj = model.get_model().mm_projector
        try:
            proj_device = next(
                p.device for p in proj.parameters() if not p.is_meta
            )
        except StopIteration:
            proj_device = vt_device

        image_features = image_features.to(device=proj_device, dtype=torch.float16)

        # Gọi class-level forward của mm_projector để bypass hook
        with torch.no_grad():
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
