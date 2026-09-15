from ReAgentV_utils.model_inference.model_inference import *
from ReAgentV_utils.tools.tool_selection import select_tools_from_question
import yaml
import os
import torch, json, os
from ReAgentV_utils.tools.scene_graph_tools.filter_keywords import filter_keywords
from ReAgentV_utils.tools.ocr_tools.rag_retriever_dynamic import retrieve_documents_with_dynamic
import numpy as np
from ReAgentV_utils.tools.scene_graph_tools.det_utils import (
    calculate_xmax_ymax,
    calculate_spatial_relations,
    relation_to_text,
    generate_scene_graph_description,
    get_det_docs,
    det_preprocess
)

from ReAgentV_utils.tools.audio_tools.asr_utils import get_asr_docs
from ReAgentV_utils.tools.ocr_tools.ocr_utils import get_ocr_docs


def retrieve_modal_info(video_path, text, frames, raw_video, clip_model, clip_processor, ocr_docs_total=None, asr_docs_total=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(base_dir, "ReAgentV_config", "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    rag_threshold  = cfg.get("rag_threshold", 0.3)
    clip_threshold = cfg.get("clip_threshold", 0.3)
    beta           = cfg.get("beta", 3.0)
    max_frames_num = cfg.get("max_frames_num", 64)
    
    
    qs = ""
    USE_OCR = USE_ASR = USE_DET = False
    
    res = select_tools_from_question(text)
    USE_OCR = res.get("USE_OCR")
    USE_ASR = res.get("USE_ASR")
    USE_DET = res.get("USE_DET")

    if USE_DET:
        # Xử lý theo mini-batch để không bao giờ bị tràn VRAM
        try:
            clip_device = next(clip_model.parameters()).device
        except Exception:
            clip_device = getattr(clip_model, "device", torch.device("cuda:0"))
        clip_features_list = []
        batch_size = 16
        with torch.no_grad():
            for i in range(0, len(raw_video), batch_size):
                batch_frames = raw_video[i:i + batch_size]
                processed = clip_processor(images=batch_frames, return_tensors="pt")["pixel_values"].to(clip_device, dtype=torch.float16)
                feats = clip_model.get_image_features(processed)
                clip_features_list.append(feats)
        clip_img_feats = torch.cat(clip_features_list, dim=0)

    if USE_OCR:
        ocr_docs_total = get_ocr_docs(frames)

    if USE_ASR:
        txt_path = os.path.join("restore/audio", os.path.basename(video_path).split(".")[0] + ".txt")
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                asr_docs_total = f.readlines()
        else:
            audio_path = txt_path.replace(".txt", ".wav")
            asr_docs_total = get_asr_docs(video_path, audio_path)
            with open(txt_path, 'w', encoding='utf-8') as f:
                for doc in asr_docs_total:
                    f.write(doc + '\n')

    det_docs, ocr_docs, asr_docs = [], [], []
    det_top_idx = []
    cot_json_instruction = """{
        "ASR": Optional[str]. The subtitles of the video that may be relevant to the question you want to retrieve, in two sentences. If you do not need this information, please return null.
        "DET": Optional[list]. (The output must include only physical entities, not abstract concepts, less than five entities.) All the physical entities and their locations related to the question you want to retrieve, not abstract concepts. If you do not need this information, please return null.
        "TYPE": Optional[list]. (The output must be specified as null or a list containing only one or more of the following strings: "location", "number", "relation". No other values are valid for this field.) The information you want to obtain about the detected objects. If you need the object location in the video frame, output "location"; if you need the number of a specific object, output "number"; if you need the positional relationship between objects, output "relation".
    }
    
    ## Example 1:
    Question: How many blue balloons are over the long table in the middle of the room at the end of this video? A. 1. B. 2. C. 3. D. 4.
    Your retrieve can be:
    {
        "ASR": "The location and the color of balloons, the number of the blue balloons.",
        "DET": ["blue balloons", "long table"],
        "TYPE": ["relation", "number"]
    }
    
    ## Example 2:
    Question: In the lower left corner of the video, what color is the woman wearing on the right side of the man in black clothes? A. Blue. B. White. C. Red. D. Yellow.
    Your retrieve can be:
    {
        "ASR": null,
        "DET": ["the man in black", "woman"],
        "TYPE": ["location", "relation"]
    }
    
    ## Example 3:
    Question: In which country is the comedy featured in the video recognized worldwide? A. China. B. UK. C. Germany. D. United States.
    Your retrieve can be:
    {
        "ASR": "The country recognized worldwide for its comedy.",
        "DET": null,
        "TYPE": null
    }
    
    Note that you don't need to answer the question in this step, so you don't need any information about the video or image. You only need to provide your retrieve request (it's optional), and I will help you retrieve the information you want. Please provide the JSON format.
    """
    
    retrieve_pmt = f"Question: {text}\n\n" \
                   "To answer the question step by step, you can provide your retrieve request to assist you by the following JSON format:\n" \
                   f"{cot_json_instruction}"
    try:
        json_request = llava_inference(retrieve_pmt, None)
        import re
        cleaned = re.sub(r"^```(?:json)?\s*", "", json_request.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()
        try:
            request_json = json.loads(cleaned)
        except Exception:
            m = re.search(r"\{.*?\}", json_request, re.DOTALL)
            request_json = json.loads(m.group(0)) if m else {}
    except Exception:
        request_json = {}

    query = [text]

    # DET
    if USE_DET:
        try:
            request_det = request_json.get("DET", None)
            request_det = filter_keywords(request_det)
            clip_text = ["A picture of " + txt for txt in request_det] if request_det else ["A picture of object"]
        except Exception:
            clip_text = ["A picture of object"]

        try:
            clip_device = next(clip_model.parameters()).device
        except Exception:
            clip_device = getattr(clip_model, "device", torch.device("cuda:0"))

        clip_inputs = clip_processor(text=clip_text, return_tensors="pt", padding=True, truncation=True)
        clip_inputs = {k: v.to(clip_device) for k, v in clip_inputs.items()}

        with torch.no_grad():
            text_features = clip_model.get_text_features(**clip_inputs)
            sim_matrix = clip_img_feats @ text_features.T
            if sim_matrix.dim() == 1:
                similarities = sim_matrix.cpu()
            else:
                similarities = sim_matrix.mean(dim=-1).cpu()
            similarities = np.array(similarities, dtype=np.float64)
            sim_sum = np.sum(similarities)
            alpha = beta * (len(similarities) / 16)
            if sim_sum > 0:
                similarities = similarities * alpha / sim_sum

        del clip_inputs, clip_img_feats, text_features
        torch.cuda.empty_cache()

        det_top_idx = [idx for idx in range(len(similarities)) if similarities[idx] > clip_threshold]
        selected_frames = [frames[i] for i in det_top_idx]


        if request_det:
            det_raw_docs = get_det_docs(selected_frames, request_det)
            L = "location" in request_json.get("TYPE", [])
            R = "relation" in request_json.get("TYPE", [])
            N = "number" in request_json.get("TYPE", [])
            det_docs = det_preprocess(det_raw_docs, location=L, relation=R, number=N)

    # OCR
    if USE_OCR and ocr_docs_total:
        ocr_query = query + request_json.get("DET", [])
        ocr_docs, _ = retrieve_documents_with_dynamic(ocr_docs_total, ocr_query, threshold=rag_threshold)

    # ASR
    if USE_ASR and asr_docs_total:
        asr_query = query
        if request_json.get("ASR"):
            asr_query.append(request_json["ASR"])
        asr_docs, _ = retrieve_documents_with_dynamic(asr_docs_total, asr_query, threshold=rag_threshold)

    modal_info = {"OCR": ocr_docs, "ASR": asr_docs, "DET": det_docs}
    return modal_info, det_top_idx, USE_OCR, USE_ASR, USE_DET
