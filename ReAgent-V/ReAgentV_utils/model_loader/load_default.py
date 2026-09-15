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

    # Load LLaVA (other open-source models can also be used)
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
    
    # Đảm bảo vision_tower và mm_projector không dính meta tensor hoặc hook lỗi của accelerate
    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        try:
            from transformers import SigLipVisionModel
            vt_name = getattr(vision_tower, "vision_tower_name", "google/siglip-so400m-patch14-384")
            vt_model = SigLipVisionModel.from_pretrained(
                vt_name,
                torch_dtype=torch_dtype,
                device_map="cuda:0"
            )
            del vt_model.vision_model.encoder.layers[-1:]
            import torch.nn as nn
            vt_model.vision_model.head = nn.Identity()
            vt_model.requires_grad_(False)
            vision_tower.vision_tower = vt_model
            vision_tower.is_loaded = True
            
            # Gỡ bỏ hook của accelerate trên vision_tower và model.model.vision_tower
            from accelerate.hooks import remove_hook_from_module
            for m in [vision_tower, getattr(model.get_model(), "vision_tower", None)]:
                if m is not None:
                    try:
                        remove_hook_from_module(m, recurse=True)
                    except Exception:
                        pass
        except Exception as e:
            print(f"Warning re-loading vision tower: {e}")

    # Patch model.encode_images để tránh accelerate hook chặn forward trên vision_tower
    raw_encode_images = model.encode_images
    def safe_encode_images(images):
        vt = model.get_model().get_vision_tower()
        # Đảm bảo images cùng device và dtype với vision_tower
        vt_device = next(vt.parameters()).device
        vt_dtype = next(vt.parameters()).dtype
        images = images.to(device=vt_device, dtype=vt_dtype)
        # Gọi thẳng vision_tower mà không qua hook của accelerate
        image_features = vt(images)
        proj = model.get_model().mm_projector
        proj_device = next(proj.parameters()).device
        proj_dtype = next(proj.parameters()).dtype
        image_features = image_features.to(device=proj_device, dtype=proj_dtype)
        image_features = proj(image_features)
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
