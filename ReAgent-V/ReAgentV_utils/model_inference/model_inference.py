import copy
import torch
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
tokenizer = None
model = None


def llava_inference(qs, video):
    if video is not None:
        question = DEFAULT_IMAGE_TOKEN + qs
    else:
        question = qs
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    # Tìm device CUDA thật sự (không phải meta) để đặt input_ids
    try:
        target_device = next(
            p.device for p in model.parameters()
            if not p.is_meta and p.device.type == "cuda"
        )
    except StopIteration:
        target_device = torch.device("cuda:0")
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(target_device)
    
    if video is not None:
        cont = model.generate(
            input_ids,
            images=video,
            modalities=["video"],
            do_sample=False,
            max_new_tokens=4096,
            num_beams=1
        )
    else:
        cont = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=4096,
        )
    
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    return text_outputs
