import os
import json
import pickle
import socket
import ast
import copy
from typing import List, Optional, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from PIL import Image
from string import Template
from decord import VideoReader, cpu

from ReAgentV_utils.model_inference.model_inference import tokenizer as _tokenizer, model as _model
from ReAgentV_utils.prompt_builder.prompt import (
    tool_retrieval_prompt_template,
    eval_reward_prompt_template,
    conservative_template_str,
    neutral_template_str,
    aggressive_template_str,
    meta_agent_prompt_template
)
from ReAgentV_utils.prompt_builder.covr_prompt import (
    covr_rerank_prompt_template,
    covr_critic_prompt_template,
    covr_query_expansion_template,
    covr_feedback_refinement_template,
    covr_reason_target_prompt_template,
    covr_pairwise_tournament_template,
)
from ReAgentV_utils.frame_selection_ecrs.ECRS_frame_selection import select_keyframes
from ReAgentV_utils.video_processor.process_video import load_video_frames
from ReAgentV_utils.video_processor.covr_loader import extract_middle_frame
from ReAgentV_utils.prompt_builder.build_multimodal_prompt import build_multimodal_prompt
from ReAgentV_utils.model_loader.load_default import load_default
from ReAgentV_utils.model_inference.model_inference import llava_inference
from ReAgentV_utils.critical_question_generator.generate_critical_question import evaluate_answer
from ReAgentV_utils.memory.memory_bank import ToolMemoryBank, _extract_scalar_reward, _strip_llava_output


class ReAgentV:
    def __init__(
        self,
        clip_model,
        clip_processor,
        whisper_model,
        whisper_processor,
        tokenizer,
        model,
        image_processor,
        conv_template,
    ):
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.whisper_model = whisper_model
        self.whisper_processor = whisper_processor
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.conv_template = conv_template

        _tokenizer = tokenizer
        _model = model

    @classmethod
    def load_default(cls, path_dict: dict):
        (
            clip_model,
            clip_processor,
            whisper_model,
            whisper_processor,
            tokenizer,
            model,
            image_processor,
            conv_template,
        ) = (
            lambda m: (
                m["clip_model"],
                m["clip_processor"],
                m["whisper_model"],
                m["whisper_processor"],
                m["tokenizer"],
                m["model"],
                m["image_processor"],
                m["conv_template"],
            )
        )(modules := load_default(path_dict))
        if not hasattr(model, "hf_device_map"):
            model = model.to("cuda")

        from ReAgentV_utils.model_inference import model_inference

        model_inference.tokenizer = tokenizer
        model_inference.model = model

        return cls(
            clip_model,
            clip_processor,
            whisper_model,
            whisper_processor,
            tokenizer,
            model,
            image_processor,
            conv_template,
        )

    def load_and_sample_video(self, question: str, video_path: str, sample_rate: int = 1, force_sample: bool = False):

        frames = load_video_frames(video_path, sample_rate, force_sample)
        key_frames, key_indices = select_keyframes(frames, question, self.clip_model, self.clip_processor)
        max_frames_num = len(key_frames)
        raw_video = [f for f in frames]

        # Tìm device CUDA thật sự (không phải meta) từ model parameters
        # Bất kể vision_tower hay LLM đặt ở đâu, luôn ưu tiên cuda:0
        try:
            dev = next(
                p.device for p in self.model.parameters()
                if not p.is_meta and p.device.type == "cuda"
            )
        except StopIteration:
            dev = torch.device("cuda:0")

        video_tensor = (
            self.image_processor.preprocess(key_frames, return_tensors="pt")["pixel_values"]
            .to(dev, dtype=torch.float16)
        )
        video_for_model = [video_tensor]

        return frames, key_frames, key_indices, max_frames_num, raw_video, video_for_model

    def retrieve_modal_info(
        self,
        video_path: str,
        question: str,
        frames: list,
        raw_video: list,
        clip_model,
        clip_processor,
    ):
        from ReAgentV_utils.tools.extract_modal_info import retrieve_modal_info
        return retrieve_modal_info(
            video_path=video_path,
            text=question,
            frames=frames,
            raw_video=raw_video,
            clip_model=clip_model,
            clip_processor=clip_processor,
        )

    def build_multimodal_prompt(self, question: str, modal_info: dict, det_top_idx: list, max_frames_num: int, USE_DET: bool, USE_ASR: bool, USE_OCR: bool) -> str:
       
        return build_multimodal_prompt(
            text=question,
            modal_info=modal_info,
            det_top_idx=det_top_idx,
            max_frames_num=max_frames_num,
            USE_DET=USE_DET,
            USE_ASR=USE_ASR,
            USE_OCR=USE_OCR,
        )

    def generate_critical_questions(self, question: str, initial_answer: str, context_info: dict, video) -> list[str]:
       
        return evaluate_answer(question=question, answer=initial_answer, context_info=context_info, video=video)

    def generate_eval_report(self, question: str, initial_answer: str, context_info, video) -> str:
       
        if isinstance(context_info, str):
            context_str = context_info
        else:
            context_str = json.dumps(context_info, indent=2)
        critic_prompt = eval_reward_prompt_template.format(
            question=question,
            context=context_str,
            initial_answer=initial_answer,
        )
        eval_report = llava_inference(critic_prompt, video)
        return eval_report

    def get_reflective_final_answer(
        self,
        question: str,
        initial_answer: str,
        eval_report: str,
        video,
    ) -> tuple[str, dict]:
       
        neutral_template = Template(neutral_template_str).substitute(
            text=question, answer=initial_answer, eval_report=eval_report
        )
        neutral_res = llava_inference(neutral_template, video)
        def _parse_agent_json(res_text, default_ans):
            if not res_text:
                return default_ans, 0.0
            import re
            try:
                data = json.loads(res_text.strip())
                return data.get("final_answer", default_ans), float(data.get("confidence", 0.5))
            except Exception:
                pass
            try:
                cleaned = re.sub(r"^```(?:json)?\s*", "", res_text.strip(), flags=re.MULTILINE)
                cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()
                data = json.loads(cleaned)
                return data.get("final_answer", default_ans), float(data.get("confidence", 0.5))
            except Exception:
                pass
            try:
                ans_m = re.search(r'"final_answer"\s*:\s*"([^"]+)"', res_text)
                conf_m = re.search(r'"confidence"\s*:\s*([0-9.]+)', res_text)
                ans = ans_m.group(1) if ans_m else default_ans
                conf = float(conf_m.group(1)) if conf_m else 0.5
                return ans, conf
            except Exception:
                return default_ans, 0.5

        ans_neutral, conf_neutral = _parse_agent_json(neutral_res, initial_answer)

        aggressive_template = Template(aggressive_template_str).substitute(
            text=question, answer=initial_answer, eval_report=eval_report
        )
        aggressive_res = llava_inference(aggressive_template, video)
        ans_aggressive, conf_aggressive = _parse_agent_json(aggressive_res, initial_answer)

        conservative_template = Template(conservative_template_str).substitute(
            text=question, answer=initial_answer, eval_report=eval_report
        )
        conservative_res = llava_inference(conservative_template, video)
        ans_conservative, conf_conservative = _parse_agent_json(conservative_res, initial_answer)

        meta_template = Template(meta_agent_prompt_template).substitute(
            answer_conservative=ans_conservative,
            conf_conservative=conf_conservative,
            answer_neutral=ans_neutral,
            conf_neutral=conf_neutral,
            answer_aggressive=ans_aggressive,
            conf_aggressive=conf_aggressive,
            text=question,
            initial_answer=initial_answer,
        )
        final_model_answer = llava_inference(meta_template, video)

        confidences = {
            "neutral": conf_neutral,
            "aggressive": conf_aggressive,
            "conservative": conf_conservative,
        }
        return final_model_answer

    # ===================================================================
    # CoVR: Composed Video Retrieval Methods
    # ===================================================================

    def load_corpus_index(
        self, index_path: str, video_base_dir: Optional[str] = None
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Load the pre-computed CLIP corpus index built by covr_indexer.py.

        Args:
            index_path     : Path to covr_corpus_index.pt.
            video_base_dir : Optional root directory of video dataset. If provided or if
                             the indexed paths do not exist, paths are automatically
                             remapped to match video_base_dir or existing files on disk.

        Returns:
            embeddings  : Tensor [N, D] of L2-normalised CLIP image features.
            video_paths : List of N absolute video file paths.
        """
        data = torch.load(index_path, map_location="cpu")
        embeddings  = data["embeddings"]   # [N, D]
        video_paths = list(data["video_paths"])  # List[str]

        # Auto-remap paths if video_base_dir is specified or if paths are missing
        if video_paths and (video_base_dir or not os.path.exists(video_paths[0])):
            remapped_paths = []
            for p in video_paths:
                norm_p = p.replace("\\", "/")
                # Extract relative video path e.g. "subfolder/video_id.mp4"
                parts = norm_p.split("/")
                rel = "/".join(parts[-2:]) if len(parts) >= 2 else os.path.basename(norm_p)
                
                if video_base_dir:
                    candidate = os.path.join(video_base_dir, rel)
                else:
                    candidate = p

                if not os.path.exists(candidate) and video_base_dir:
                    # Fallback check if rel matches direct filename
                    candidate_direct = os.path.join(video_base_dir, os.path.basename(norm_p))
                    if os.path.exists(candidate_direct):
                        candidate = candidate_direct

                remapped_paths.append(candidate)
            video_paths = remapped_paths

        print(f"[ReAgentV] Corpus index loaded: {len(video_paths):,} videos.")
        return embeddings, video_paths

    def simulate_target_description(
        self,
        query_image: Image.Image,
        query_text: str,
    ) -> str:
        """
        Reason-then-Retrieve Stage 0:
        Prompt LLaVA-Video with the reference image and edit instruction to imagine
        and describe the resulting TARGET VIDEO.

        Returns:
            A concise visual description (str) of the expected target video.
        """
        try:
            dev = next(
                p.device for p in self.model.parameters()
                if not p.is_meta and p.device.type == "cuda"
            )
        except StopIteration:
            dev = torch.device("cuda:0")

        # Encode single reference image frame as visual input for LLaVA
        q_img_tensor = (
            self.image_processor.preprocess([query_image], return_tensors="pt")["pixel_values"]
            .to(dev, dtype=torch.float16)
        )  # [1, C, H, W]
        # LLaVA video modality expects list of frame tensors
        vision_input = [q_img_tensor]

        prompt = covr_reason_target_prompt_template.format(edit_prompt=query_text)
        try:
            raw_output = llava_inference(prompt, vision_input, max_new_tokens=256)
            cleaned = _strip_llava_output(raw_output)
            
            # Parse JSON
            try:
                data = json.loads(cleaned)
                if "target_video_keywords" in data:
                    keywords = data["target_video_keywords"]
                    if isinstance(keywords, list):
                        target_desc = ", ".join(keywords)
                    else:
                        target_desc = str(keywords)
                    if target_desc:
                        print(f"[ReAgentV Reason] Target simulation: '{target_desc}'")
                        return target_desc
                target_desc = data.get("target_video_description", "").strip()
                if target_desc:
                    print(f"[ReAgentV Reason] Target simulation: '{target_desc}'")
                    return target_desc
            except json.JSONDecodeError:
                # Regex fallback
                import re
                m = re.search(r'"?target_video_description"?\s*:\s*"([^"]+)"', cleaned, re.IGNORECASE)
                if m:
                    target_desc = m.group(1).strip()
                    print(f"[ReAgentV Reason] Target simulation (Regex): '{target_desc}'")
                    return target_desc

        except Exception as e:
            print(f"[ReAgentV Reason] simulate_target_description failed: {e}")

        # Fallback to query_text
        return query_text

    def encode_covr_query(
        self,
        query_image: Image.Image,
        query_text: str,
        alpha: float = 0.5,
        query_expansion_hint: str = "",
        reasoned_description: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Encode a composed query into a CLIP embedding.
        If reasoned_description is provided (Reason-then-Retrieve mode), the text
        features will focus heavily on the target scene description.

        Args:
            query_image           : PIL Image (middle frame of reference video).
            query_text            : Edit instruction / prompt text.
            alpha                 : Weight for image feature (1-alpha goes to text).
            query_expansion_hint  : Optional extra keywords appended to query_text.
            reasoned_description : Optional pre-simulated target video description from LLaVA.

        Returns:
            query_feat : Tensor [1, D] L2-normalised composed query embedding.
        """
        try:
            clip_device = next(self.clip_model.parameters()).device
        except Exception:
            clip_device = torch.device("cuda:0")

        # Select text to encode: prioritized reasoned_description if available
        if reasoned_description:
            active_text = reasoned_description.strip()
        else:
            active_text = query_text.strip()

        # Format text with context prefix to align with target video representations
        if not active_text.lower().startswith("a video of") and not active_text.lower().startswith("a video showing"):
            base_text = f"a video showing {active_text}"
        else:
            base_text = active_text

        if query_expansion_hint:
            full_text = f"{base_text}, {query_expansion_hint.strip()}"
        else:
            full_text = base_text

        with torch.no_grad():
            # Image branch
            img_inputs = self.clip_processor(
                images=query_image, return_tensors="pt"
            )["pixel_values"].to(clip_device, dtype=torch.float16)
            img_feat = self.clip_model.get_image_features(img_inputs)  # [1, D]
            img_feat = F.normalize(img_feat.float(), dim=-1)

            # Text branch
            txt_inputs = self.clip_processor(
                text=[full_text], return_tensors="pt",
                padding=True, truncation=True, max_length=77,
            )
            txt_inputs = {k: v.to(clip_device) for k, v in txt_inputs.items()}
            txt_feat = self.clip_model.get_text_features(**txt_inputs)   # [1, D]
            txt_feat = F.normalize(txt_feat.float(), dim=-1)

        # Target-driven Composed Visual Query Encoding:
        # In Reason-then-Retrieve mode, reasoned_description provides the simulated future target scene
        # (e.g. 'yellow tulips in a field' or 'goats on a hillside').
        # eff_alpha MUST remain low (<= 0.15) so the unedited source image does not overpower the target description
        # and pull distractors (videos matching the unedited state) into coarse top-1.
        if reasoned_description:
            eff_alpha = min(alpha, 0.15)
        else:
            eff_alpha = alpha

        # Compose and re-normalise
        query_feat = F.normalize(
            eff_alpha * img_feat + (1.0 - eff_alpha) * txt_feat, dim=-1
        )  # [1, D]
        return query_feat.cpu()

    def coarse_search(
        self,
        query_image: Image.Image,
        query_text: str,
        corpus_embeddings: torch.Tensor,
        corpus_paths: List[str],
        top_n: int = 20,
        alpha: float = 0.5,
        query_expansion_hint: str = "",
        exclude_paths: Optional[set] = None,
        reasoned_description: Optional[str] = None,
        candidate_pool_size: int = 50,
    ) -> List[Tuple[str, float]]:
        """
        Stage 1 — Cascade Candidate Generation & Fast Pre-ranking:
        1. Fast Broad Pool: Retrieves Top-M (candidate_pool_size=50) candidates using
           the composed query vector (reasoned target state + visual features) across the full corpus.
        2. Fast Intent-Preserving Pre-ranking: Focuses candidate filtering strictly on
           the target transformation state without penalizing modified visual entities.
        3. Returns Top-N optimal candidates for deep LLaVA reranking.
        """
        try:
            clip_device = next(self.clip_model.parameters()).device
        except Exception:
            clip_device = torch.device("cuda:0")

        # Encode full composed query
        query_feat = self.encode_covr_query(
            query_image, query_text, alpha, query_expansion_hint, reasoned_description=reasoned_description
        )  # [1, D]

        # Module 3: Negative Vector Repulsion (Rocchio Feedback)
        # Shift search center away from clusters of rejected candidates
        if exclude_paths:
            path_to_idx = {p: i for i, p in enumerate(corpus_paths)}
            q_adapted = query_feat.clone().to(device=corpus_embeddings.device, dtype=torch.float32)
            repelled_count = 0
            for neg_path in exclude_paths:
                if neg_path in path_to_idx:
                    neg_idx = path_to_idx[neg_path]
                    neg_feat = corpus_embeddings[neg_idx:neg_idx+1].to(device=q_adapted.device, dtype=torch.float32)
                    q_adapted = q_adapted - 0.20 * neg_feat
                    repelled_count += 1
            if repelled_count > 0:
                query_feat = F.normalize(q_adapted, p=2, dim=-1)
                print(f"[ReAgentV Rocchio] Applied negative vector repulsion for {repelled_count} excluded candidates.")
            else:
                query_feat = query_feat.to(device=corpus_embeddings.device, dtype=torch.float32)
        else:
            query_feat = query_feat.to(device=corpus_embeddings.device, dtype=torch.float32)

        # Fast GPU/CPU cosine similarity against entire corpus [N]
        sims_composed = (corpus_embeddings.to(dtype=torch.float32) @ query_feat.T).squeeze(-1)  # [N]

        # VRAgent-inspired Target Scene Alignment:
        # If reasoned_description is present, also compute direct text-to-video alignment
        # to ensure new target entities (goats, horses, Mars) strongly lead candidate selection.
        if reasoned_description:
            with torch.no_grad():
                target_text = f"a video showing {reasoned_description.strip()}"
                target_inputs = self.clip_processor(
                    text=[target_text], return_tensors="pt",
                    padding=True, truncation=True, max_length=77,
                )
                target_inputs = {k: v.to(clip_device) for k, v in target_inputs.items()}
                target_feat = self.clip_model.get_text_features(**target_inputs)
                target_feat = F.normalize(target_feat.float(), dim=-1).to(device=corpus_embeddings.device)

            sims_target = (corpus_embeddings.to(dtype=torch.float32) @ target_feat.T).squeeze(-1)
            sims_final = 0.40 * sims_target + 0.60 * sims_composed
        else:
            sims_final = sims_composed

        # Exclude previously rejected candidates
        if exclude_paths:
            for i, vp in enumerate(corpus_paths):
                if vp in exclude_paths:
                    sims_final[i] = -1.0

        # Step 1: Broad Candidate Pool (Top-M, e.g. 50-100 videos)
        pool_m = min(max(candidate_pool_size, top_n), len(corpus_paths))
        pool_indices = torch.topk(sims_final, k=pool_m).indices

        # Step 2: Extract top-N candidate paths and scores
        top_sub_indices = pool_indices[:top_n].tolist()
        results = [(corpus_paths[idx], float(sims_final[idx])) for idx in top_sub_indices]
        return results

    def _video_to_tensor(
        self,
        video_path: str,
        tensor_cache: Optional[Dict[str, List[torch.Tensor]]] = None,
    ) -> Optional[List[torch.Tensor]]:
        """
        Load a candidate video and convert it to the video tensor format
        expected by LLaVA (same as load_and_sample_video output).
        Returns None if the video cannot be loaded.
        Reuses tensor_cache if available to eliminate redundant decodes.
        """
        if tensor_cache is not None and video_path in tensor_cache:
            return tensor_cache[video_path]

        try:
            # num_threads=0 automatically utilizes all available CPU cores for maximum decode speed
            vr = VideoReader(video_path, ctx=cpu(), num_threads=0)
            total_frames = len(vr)
            if total_frames == 0:
                return None
            # Uniformly sample 4 keyframes across the video duration for temporal awareness
            num_samples = min(4, total_frames)
            indices = np.linspace(0, total_frames - 1, num_samples, dtype=int).tolist()
            sampled_np = vr.get_batch(indices).asnumpy()
            key_frames = [Image.fromarray(f) for f in sampled_np]

            try:
                dev = next(
                    p.device for p in self.model.parameters()
                    if not p.is_meta and p.device.type == "cuda"
                )
            except StopIteration:
                dev = torch.device("cuda:0")
            vid_tensor = (
                self.image_processor.preprocess(key_frames, return_tensors="pt")["pixel_values"]
                .to(dev, dtype=torch.float16)
            )
            res = [vid_tensor]
            if tensor_cache is not None:
                tensor_cache[video_path] = res
            return res
        except Exception as e:
            print(f"[ReAgentV] _video_to_tensor failed for {video_path}: {e}")
            return None

    def pairwise_tie_break(
        self,
        query_image: Image.Image,
        query_text: str,
        cand_a_path: str,
        cand_b_path: str,
        tensor_cache: Optional[Dict[str, List[torch.Tensor]]] = None,
        clip_rank_a: int = 99,
        clip_rank_b: int = 99,
        rel_a: float = 0.0,
        rel_b: float = 0.0,
    ) -> Tuple[str, float, str]:
        """
        VRAgent-inspired Pairwise Tournament Tie-Breaker.
        Directly compares Candidate A and Candidate B head-to-head in a single prompt
        (1 query image + 2 frames from A + 2 frames from B = 5 frames) to break ties
        between near-identical sister clips.

        Visual Input Layout:
        - Frame 1: Reference Image (starting state)
        - Frames 2, 3: Candidate Video A keyframes
        - Frames 4, 5: Candidate Video B keyframes

        Returns:
            (winner, confidence, reason) where winner in {"A", "B"}
        """
        import re

        try:
            dev = next(
                p.device for p in self.model.parameters()
                if not p.is_meta and p.device.type == "cuda"
            )
        except StopIteration:
            dev = torch.device("cuda:0")

        q_img_tensor = (
            self.image_processor.preprocess([query_image], return_tensors="pt")["pixel_values"]
            .to(dev, dtype=torch.float16)
        )

        vid_a_tensor = self._video_to_tensor(cand_a_path, tensor_cache=tensor_cache)
        vid_b_tensor = self._video_to_tensor(cand_b_path, tensor_cache=tensor_cache)

        if vid_a_tensor is None or vid_b_tensor is None:
            del q_img_tensor
            return "A", 0.5, "Failed to load candidate video tensors"

        # Extract 1 representative keyframe each (the last frame usually shows the completed edit)
        idx_a = vid_a_tensor[0].shape[0] - 1
        idx_b = vid_b_tensor[0].shape[0] - 1
        frame_a = vid_a_tensor[0][idx_a:idx_a+1].to(dev, dtype=torch.float16)
        frame_b = vid_b_tensor[0][idx_b:idx_b+1].to(dev, dtype=torch.float16)

        # --- Symmetric Debiased Tournament with Incumbent Context Shield ---
        prompt = covr_pairwise_tournament_template.format(edit_prompt=query_text)

        def _extract_tournament_json(text: str) -> Optional[dict]:
            if not text:
                return None
            c = _strip_llava_output(text)
            m = re.search(r'\{.*\}', c, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
            try:
                return json.loads(c)
            except Exception:
                return None

        # Forward pass 1: Frame 2 is Candidate A (Incumbent), Frame 3 is Candidate B (Challenger)
        combined_tensor_ab = torch.cat([q_img_tensor, frame_a, frame_b], dim=0)
        pref_1, conf_1, reason_1 = "A", 0.5, ""
        s_edit_a_1, s_edit_b_1 = 0.5, 0.5
        s_pres_a_1, s_pres_b_1 = 0.5, 0.5

        try:
            raw_output_1 = llava_inference(prompt, [combined_tensor_ab], max_new_tokens=256)
            data_1 = _extract_tournament_json(raw_output_1)
            if data_1 and isinstance(data_1, dict):
                pref_1 = str(data_1.get("preferred", "A")).strip().upper()
                conf_1 = float(data_1.get("confidence", 0.5))
                reason_1 = str(data_1.get("reason", ""))
                s_edit_a_1 = float(data_1.get("s_edit_a", 0.5))
                s_edit_b_1 = float(data_1.get("s_edit_b", 0.5))
                s_pres_a_1 = float(data_1.get("s_preservation_a", 0.5))
                s_pres_b_1 = float(data_1.get("s_preservation_b", 0.5))
            else:
                raise ValueError("Regex fallback required")
        except Exception:
            m_pref = re.search(r'"?preferred"?\s*:\s*"?.*?\b(A|B)\b"?', raw_output_1 if 'raw_output_1' in locals() else "", re.IGNORECASE)
            m_conf = re.search(r'"?confidence"?\s*:\s*([0-9]*\.?[0-9]+)', raw_output_1 if 'raw_output_1' in locals() else "")
            m_ea = re.search(r'"?s_edit_a"?\s*:\s*([0-9]*\.?[0-9]+)', raw_output_1 if 'raw_output_1' in locals() else "")
            m_eb = re.search(r'"?s_edit_b"?\s*:\s*([0-9]*\.?[0-9]+)', raw_output_1 if 'raw_output_1' in locals() else "")
            m_pa = re.search(r'"?s_preservation_a"?\s*:\s*([0-9]*\.?[0-9]+)', raw_output_1 if 'raw_output_1' in locals() else "")
            m_pb = re.search(r'"?s_preservation_b"?\s*:\s*([0-9]*\.?[0-9]+)', raw_output_1 if 'raw_output_1' in locals() else "")
            pref_1 = m_pref.group(1).upper() if m_pref else "A"
            conf_1 = float(m_conf.group(1)) if m_conf else 0.5
            s_edit_a_1 = float(m_ea.group(1)) if m_ea else 0.5
            s_edit_b_1 = float(m_eb.group(1)) if m_eb else 0.5
            s_pres_a_1 = float(m_pa.group(1)) if m_pa else 0.5
            s_pres_b_1 = float(m_pb.group(1)) if m_pb else 0.5
            reason_1 = "Parsed via fallback regex"
        del combined_tensor_ab

        # Forward pass 2: Inverted order (B as Video A in prompt, A as Video B in prompt)
        combined_tensor_ba = torch.cat([q_img_tensor, frame_b, frame_a], dim=0)
        pref_2, conf_2, reason_2 = "B", 0.5, ""
        s_edit_a_2, s_edit_b_2 = 0.5, 0.5
        s_pres_a_2, s_pres_b_2 = 0.5, 0.5

        try:
            raw_output_2 = llava_inference(prompt, [combined_tensor_ba], max_new_tokens=256)
            data_2 = _extract_tournament_json(raw_output_2)
            if data_2 and isinstance(data_2, dict):
                # In pass 2: prompt Video A is Cand B, prompt Video B is Cand A
                raw_pref_2 = str(data_2.get("preferred", "A")).strip().upper()
                pref_2 = "B" if raw_pref_2 == "A" else "A"
                conf_2 = float(data_2.get("confidence", 0.5))
                reason_2 = str(data_2.get("reason", ""))
                s_edit_b_2 = float(data_2.get("s_edit_a", 0.5))
                s_edit_a_2 = float(data_2.get("s_edit_b", 0.5))
                s_pres_b_2 = float(data_2.get("s_preservation_a", 0.5))
                s_pres_a_2 = float(data_2.get("s_preservation_b", 0.5))
            else:
                raise ValueError("Regex fallback required")
        except Exception:
            m_pref = re.search(r'"?preferred"?\s*:\s*"?.*?\b(A|B)\b"?', raw_output_2 if 'raw_output_2' in locals() else "", re.IGNORECASE)
            m_conf = re.search(r'"?confidence"?\s*:\s*([0-9]*\.?[0-9]+)', raw_output_2 if 'raw_output_2' in locals() else "")
            m_ea = re.search(r'"?s_edit_a"?\s*:\s*([0-9]*\.?[0-9]+)', raw_output_2 if 'raw_output_2' in locals() else "")
            m_eb = re.search(r'"?s_edit_b"?\s*:\s*([0-9]*\.?[0-9]+)', raw_output_2 if 'raw_output_2' in locals() else "")
            m_pa = re.search(r'"?s_preservation_a"?\s*:\s*([0-9]*\.?[0-9]+)', raw_output_2 if 'raw_output_2' in locals() else "")
            m_pb = re.search(r'"?s_preservation_b"?\s*:\s*([0-9]*\.?[0-9]+)', raw_output_2 if 'raw_output_2' in locals() else "")
            raw_pref_2 = m_pref.group(1).upper() if m_pref else "A"
            pref_2 = "B" if raw_pref_2 == "A" else "A"
            conf_2 = float(m_conf.group(1)) if m_conf else 0.5
            s_edit_b_2 = float(m_ea.group(1)) if m_ea else 0.5
            s_edit_a_2 = float(m_eb.group(1)) if m_eb else 0.5
            s_pres_b_2 = float(m_pa.group(1)) if m_pa else 0.5
            s_pres_a_2 = float(m_pb.group(1)) if m_pb else 0.5
            reason_2 = "Parsed via fallback regex"
        del combined_tensor_ba, q_img_tensor
        torch.cuda.empty_cache()

        # Compute multi-aspect averages across both orientations:
        avg_edit_a = (s_edit_a_1 + s_edit_a_2) / 2.0
        avg_edit_b = (s_edit_b_1 + s_edit_b_2) / 2.0
        avg_pres_a = (s_pres_a_1 + s_pres_a_2) / 2.0
        avg_pres_b = (s_pres_b_1 + s_pres_b_2) / 2.0
        avg_conf = (conf_1 + conf_2) / 2.0

        edit_diff = avg_edit_b - avg_edit_a
        pres_diff = avg_pres_b - avg_pres_a

        # ── Balanced Pairwise Tournament Decision Logic ──
        b_net = (0.70 * avg_edit_b + 0.30 * avg_pres_b) - (0.70 * avg_edit_a + 0.30 * avg_pres_a)

        if pref_1 == "B" and pref_2 == "B":
            # Both orientations agree Challenger B is superior
            if pres_diff < -0.08:
                preferred = "A"
                confidence = 0.80
                reason = (
                    f"Incumbent Shield: Challenger B executed edit ({avg_edit_b:.2f} vs {avg_edit_a:.2f}) "
                    f"but degraded reference scene context ({avg_pres_b:.2f} vs {avg_pres_a:.2f}). Retained A."
                )
            else:
                preferred = "B"
                confidence = avg_conf
                reason = (
                    f"Challenger B demonstrated decisive superiority ({avg_edit_b:.2f} vs {avg_edit_a:.2f}) "
                    f"and preserved context ({avg_pres_b:.2f} vs {avg_pres_a:.2f}) across both orientations."
                )
        elif pref_1 == "A" and pref_2 == "A":
            # Both orientations favored A, but check if Challenger B achieved decisive net edit advantage without context collapse
            if b_net >= 0.05 and pres_diff >= -0.04:
                preferred = "B"
                confidence = 0.75
                reason = f"Challenger B overcame positional preference with superior net edit fidelity ({b_net:+.3f})."
            else:
                preferred = "A"
                confidence = avg_conf
                reason = f"Consistent preference for Candidate A across both orientations ({reason_1})"
        else:
            # Position variance / Split decision / Dead heat:
            # Candidate A is the reigning Incumbent (Top-1).
            # To protect Ground Truth from being dethroned by noisy pointwise LLaVA scores or minor variances,
            # Challenger B must demonstrate a decisive net advantage (b_net >= 0.05) or prior coarse priority.
            if b_net >= 0.05 and pres_diff >= -0.04:
                preferred = "B"
                confidence = max(conf_1, conf_2)
                reason = f"Challenger B demonstrated clear net edit fidelity advantage ({b_net:+.3f}) despite position variance."
            elif clip_rank_b < clip_rank_a and b_net >= 0.02 and pres_diff >= -0.04:
                preferred = "B"
                confidence = 0.75
                reason = f"Challenger B has superior original CLIP coarse priority (rank {clip_rank_b+1} vs {clip_rank_a+1}) and net advantage ({b_net:+.3f})."
            else:
                preferred = "A"
                confidence = max(conf_1, conf_2)
                reason = f"Incumbent Shield: Preserved Candidate A as Top-1 against Challenger B under split decision (b_net={b_net:+.3f}, ranks {clip_rank_a+1} vs {clip_rank_b+1})."

        return preferred, confidence, reason



    def agentic_rerank(
        self,
        query_image: Image.Image,
        query_text: str,
        candidate_list: List[Tuple[str, float]],
        top_k: int = 5,
        hybrid_alpha: float = 0.55,
        score_cache: Optional[Dict[str, Tuple[float, str]]] = None,
        tensor_cache: Optional[Dict[str, List[torch.Tensor]]] = None,
        enable_tournament: bool = True,
        target_sim: Optional[str] = None,
        exclude_paths: Optional[set] = None,
    ) -> List[Tuple[str, float, str]]:
        """
        Stage 2 — Fine-grained Agentic Reranking using LLaVA + Multi-Aspect Continuous Scoring + Dynamic Hybrid Alpha + Incumbent Shield.
        """
        import re

        if score_cache is None:
            score_cache = {}

        # Encode query image into a single-frame tensor for LLaVA vision input
        try:
            dev = next(
                p.device for p in self.model.parameters()
                if not p.is_meta and p.device.type == "cuda"
            )
        except StopIteration:
            dev = torch.device("cuda:0")

        q_img_tensor = (
            self.image_processor.preprocess([query_image], return_tensors="pt")["pixel_values"]
            .to(dev, dtype=torch.float16)
        )  # [1, C, H, W]

        # ── Parallel CPU Video Prefetching: decode uncached candidate videos ahead of GPU inference ──
        uncached_paths = [
            vpath for vpath, _ in candidate_list
            if vpath not in score_cache and (tensor_cache is None or vpath not in tensor_cache)
        ]
        if uncached_paths:
            with ThreadPoolExecutor(max_workers=min(4, len(uncached_paths))) as executor:
                futures = [executor.submit(self._video_to_tensor, p, tensor_cache) for p in uncached_paths]
                for f in futures:
                    try:
                        f.result()
                    except Exception:
                        pass

        evaluated_candidates = []
        for video_path, clip_score in candidate_list:

            # ── Module 3 Hard Blacklist check ─────────────────────────────────
            if exclude_paths and video_path in exclude_paths:
                print(f"[Hard Blacklist] Skipping rejected distractor: {os.path.basename(video_path)}")
                score_cache[video_path] = (0.0, "NO_MATCH")
                evaluated_candidates.append({
                    "path": video_path,
                    "clip_score": float(clip_score),
                    "rel_score": 0.0,
                    "verdict": "NO_MATCH",
                })
                continue

            # ── Cache HIT: reuse previously computed LLaVA score ──────────────
            if video_path in score_cache:
                rel_score, verdict = score_cache[video_path]
                print(f"[Cache HIT] {os.path.basename(video_path)} rel={rel_score:.3f} ({verdict})")
                evaluated_candidates.append({
                    "path": video_path,
                    "clip_score": float(clip_score),
                    "rel_score": rel_score,
                    "verdict": verdict,
                })
                continue

            # ── Cache MISS: retrieve prefetched tensor and run LLaVA inference ──
            vid_tensor = self._video_to_tensor(video_path, tensor_cache=tensor_cache)
            if vid_tensor is None:
                score_cache[video_path] = (0.0, "NO_MATCH")
                evaluated_candidates.append({
                    "path": video_path,
                    "clip_score": float(clip_score),
                    "rel_score": 0.0,
                    "verdict": "NO_MATCH",
                })
                continue

            # Build combined vision input: [query_frame + video_frames]
            combined_tensor = torch.cat([q_img_tensor, vid_tensor[0]], dim=0)
            combined_input = [combined_tensor]

            prompt = covr_rerank_prompt_template.format(edit_prompt=query_text)

            # Robust JSON extraction with regex guard (User Note 1)
            raw_output = ""
            try:
                raw_output = llava_inference(prompt, combined_input, max_new_tokens=64)
                cleaned = _strip_llava_output(raw_output)

                data = None
                m_json = re.search(r'\{.*\}', cleaned, re.DOTALL)
                if m_json:
                    try:
                        data = json.loads(m_json.group(0))
                    except Exception:
                        data = None
                if data is None:
                    try:
                        data = json.loads(cleaned)
                    except Exception:
                        data = None

                if data is not None and isinstance(data, dict):
                    # Module 2A: Multi-Aspect Continuous Scoring (s_edit-gated)
                    s_edit = float(data.get("s_edit", -1.0))
                    s_pres = float(data.get("s_preservation", -1.0))
                    s_temp = float(data.get("s_temporal", -1.0))

                    if s_edit >= 0.0 and s_pres >= 0.0 and s_temp >= 0.0:
                        # Fix 2: s_edit-gated scoring:
                        # Clear failure (s_edit < 0.35) -> auto NO_MATCH
                        if s_edit < 0.35:
                            rel_score = round(s_edit * 0.5, 4)  # capped at 0.175 -> auto NO_MATCH
                            verdict = "NO_MATCH"
                        elif s_edit < 0.50:
                            # Moderate edit execution: PARTIAL_MATCH without being crushed to NO_MATCH
                            rel_score = round(0.70 * s_edit + 0.20 * s_pres + 0.10 * s_temp, 4)
                            verdict = "PARTIAL_MATCH"
                        else:
                            rel_score = round(0.70 * s_edit + 0.20 * s_pres + 0.10 * s_temp, 4)
                            verdict = "MATCH" if rel_score >= 0.75 else "PARTIAL_MATCH"
                    else:
                        rel_score = float(data.get("relevance_score", clip_score))
                        verdict = str(data.get("verdict", "PARTIAL_MATCH")).strip().upper()

                    va = f"s_edit={s_edit:.2f}, s_pres={s_pres:.2f}, s_temp={s_temp:.2f}"
                else:
                    raise ValueError("JSON parse failed, invoking regex fallback")
            except Exception:
                # Fallback Regex for Multi-Aspect fields
                m_edit = re.search(r'"?s_edit"?\s*:\s*([0-9]*\.?[0-9]+)', raw_output)
                m_pres = re.search(r'"?s_preservation"?\s*:\s*([0-9]*\.?[0-9]+)', raw_output)
                m_temp = re.search(r'"?s_temporal"?\s*:\s*([0-9]*\.?[0-9]+)', raw_output)
                m_rel = re.search(r'"?relevance_score"?\s*:\s*([0-9]*\.?[0-9]+)', raw_output)
                m_ver = re.search(r'"?verdict"?\s*:\s*"?([^"\s]+)"?', raw_output, re.IGNORECASE)

                if m_edit and m_pres and m_temp:
                    e = float(m_edit.group(1))
                    p = float(m_pres.group(1))
                    t = float(m_temp.group(1))
                    if e < 0.35:
                        rel_score = round(e * 0.5, 4)
                        verdict = "NO_MATCH"
                    elif e < 0.50:
                        rel_score = round(0.70 * e + 0.20 * p + 0.10 * t, 4)
                        verdict = "PARTIAL_MATCH"
                    else:
                        rel_score = round(0.70 * e + 0.20 * p + 0.10 * t, 4)
                        verdict = "MATCH" if rel_score >= 0.75 else "PARTIAL_MATCH"
                    va = f"Regex: s_edit={e:.2f}, s_pres={p:.2f}, s_temp={t:.2f}"
                elif m_rel:
                    rel_score = float(m_rel.group(1))
                    verdict = m_ver.group(1).upper() if m_ver else "PARTIAL_MATCH"
                    va = f"Regex: rel={rel_score:.2f}"
                else:
                    rel_score = clip_score * 0.5
                    verdict = "PARTIAL_MATCH"
                    va = "Fallback clip*0.5"

            # Enforce s_edit gate: low edit fidelity always means NO_MATCH
            if verdict == "NO_MATCH" or rel_score < 0.25:
                rel_score = 0.10
                verdict = "NO_MATCH"

            # Print concise attribute metrics
            print(f"[Rerank] {os.path.basename(video_path)} -> rel={rel_score:.3f} ({verdict}) | {va}")

            # Store in cache for future iterations
            score_cache[video_path] = (rel_score, verdict)
            evaluated_candidates.append({
                "path": video_path,
                "clip_score": float(clip_score),
                "rel_score": rel_score,
                "verdict": verdict,
            })
            del combined_tensor, combined_input

        # Cleanup GPU cache after pointwise scoring
        del q_img_tensor
        torch.cuda.empty_cache()

        # ── Module 2A: Dynamic Hybrid Alpha with Variance Guard (Fix 4: adjusted thresholds) ──
        valid_llava_scores = [item["rel_score"] for item in evaluated_candidates if item["verdict"] != "NO_MATCH"]
        if len(valid_llava_scores) < 2:
            effective_alpha = hybrid_alpha
            print(f"[ReAgentV Dynamic Alpha] Insufficient valid candidates ({len(valid_llava_scores)} < 2) -> alpha_eff={effective_alpha:.2f} (baseline fallback)")
        else:
            score_var = float(np.var(valid_llava_scores))
            if score_var < 0.01:
                effective_alpha = 0.60
                print(f"[ReAgentV Dynamic Alpha] Score variance={score_var:.5f} < 0.01 -> alpha_eff=0.60 (relying on CLIP)")
            else:
                effective_alpha = 0.45
                print(f"[ReAgentV Dynamic Alpha] Score variance={score_var:.5f} >= 0.01 -> alpha_eff=0.45 (balanced visual-semantic)")

        # ── Pointwise Linear Hybrid Scoring with Normalized CLIP & Safety Net ──
        reranked = []
        clip_rank_map = {vpath: rank for rank, (vpath, _) in enumerate(candidate_list)}

        # Min-max normalization for clip_score within the pool to bring it to [0.0, 1.0] scale
        clip_scores = [item["clip_score"] for item in evaluated_candidates]
        min_cs = min(clip_scores) if clip_scores else 0.0
        max_cs = max(clip_scores) if clip_scores else 1.0
        cs_range = max(max_cs - min_cs, 1e-4)

        for item in evaluated_candidates:
            path = item["path"]
            norm_clip = (item["clip_score"] - min_cs) / cs_range  # [0.0, 1.0]
            effective_rel = min(item["rel_score"] * 0.3, 0.2) if item["verdict"] == "NO_MATCH" else item["rel_score"]
            linear_hybrid = effective_alpha * norm_clip + (1.0 - effective_alpha) * effective_rel
            if item["verdict"] == "NO_MATCH":
                linear_hybrid *= 0.1

            # CLIP Safety Net Prior Protection for CLIP Top-3 candidates (when not NO_MATCH)
            original_clip_rank = clip_rank_map.get(path, 99)
            if item["verdict"] != "NO_MATCH":
                if original_clip_rank == 0:
                    linear_hybrid += 0.06  # CLIP Rank 1 Prior Protection
                    print(f"[CLIP Safety Net] Applied Rank 1 Prior Protection (+0.06): {os.path.basename(path)}")
                elif original_clip_rank in [1, 2]:
                    linear_hybrid += 0.03  # CLIP Top-3 Prior Protection
                    print(f"[CLIP Safety Net] Applied Rank {original_clip_rank+1} Prior Protection (+0.03): {os.path.basename(path)}")

            reranked.append((path, round(linear_hybrid, 4), item["verdict"]))

        # Sort by final hybrid score descending
        reranked.sort(key=lambda x: x[1], reverse=True)

        # ── Module 2B: Pairwise Tournament with Balanced Tie-Breaker for Top-2 Candidates ──
        if enable_tournament and len(reranked) >= 2:
            c1_path, score_1, verdict_1 = reranked[0]
            c2_path, score_2, verdict_2 = reranked[1]
            score_margin = score_1 - score_2

            rel_1 = score_cache.get(c1_path, (0.0, ""))[0]
            rel_2 = score_cache.get(c2_path, (0.0, ""))[0]

            # Trigger tournament when Top-1 and Top-2 are in genuine competition (close call):
            trigger_tournament = (
                (score_margin <= 0.06 and rel_1 >= 0.65 and rel_2 >= 0.65) or
                (verdict_1 == "MATCH" and verdict_2 == "MATCH" and score_margin <= 0.08)
            )


            if trigger_tournament:
                print(f"\n[ReAgentV Tournament] Top-2 tie-break triggered (margin={score_margin:.4f}):")
                print(f"  Candidate A (Incumbent): {os.path.basename(c1_path)} (score={score_1:.4f}, rel={rel_1:.2f})")
                print(f"  Candidate B (Challenger): {os.path.basename(c2_path)} (score={score_2:.4f}, rel={rel_2:.2f})")

                winner, conf, reason = self.pairwise_tie_break(
                    query_image, query_text, c1_path, c2_path, tensor_cache=tensor_cache,
                    clip_rank_a=clip_rank_map.get(c1_path, 99),
                    clip_rank_b=clip_rank_map.get(c2_path, 99),
                    rel_a=rel_1,
                    rel_b=rel_2,
                )

                if winner == "B":
                    reranked[0] = (c2_path, round(score_1 + 0.001, 4), verdict_2)
                    reranked[1] = (c1_path, round(score_2, 4), verdict_1)
                    print(f"  >>> Result: Winner is Challenger B ({os.path.basename(c2_path)}) over Incumbent A (conf={conf:.2f}). Reason: {reason}")
                else:
                    print(f"  >>> Result: Confirmed Incumbent A ({os.path.basename(c1_path)}) as Top-1 (conf={conf:.2f}). Reason: {reason}")


        return reranked[:top_k]

    def covr_critic_evaluate(
        self,
        query_image: Image.Image,
        query_text: str,
        top1_video_path: str,
        tensor_cache: Optional[Dict[str, List[torch.Tensor]]] = None,
    ) -> Tuple[float, str]:
        """
        Run the Critic Agent on the current Top-1 retrieval result.
        Returns (scalar_reward, critic_raw_json) for the Memory Bank.
        Reuses tensor_cache from Stage 2 to eliminate redundant video decode.
        """
        try:
            dev = next(
                p.device for p in self.model.parameters()
                if not p.is_meta and p.device.type == "cuda"
            )
        except StopIteration:
            dev = torch.device("cuda:0")

        q_img_tensor = (
            self.image_processor.preprocess([query_image], return_tensors="pt")["pixel_values"]
            .to(dev, dtype=torch.float16)
        )
        vid_tensor = self._video_to_tensor(top1_video_path, tensor_cache=tensor_cache)
        if vid_tensor is None:
            return 0.0, '{"scalar_reward": 0.0, "structured_feedback": "Video could not be loaded."}'

        combined = torch.cat([q_img_tensor, vid_tensor[0]], dim=0)
        combined_input = [combined]

        candidate_id = os.path.basename(top1_video_path)
        prompt = covr_critic_prompt_template.format(
            edit_prompt=query_text,
            candidate_id=candidate_id,
        )
        critic_raw = llava_inference(prompt, combined_input, max_new_tokens=128)
        scalar_reward = _extract_scalar_reward(critic_raw)

        del combined, combined_input, q_img_tensor
        torch.cuda.empty_cache()
        return scalar_reward, critic_raw

    def adaptive_covr_retrieval(
        self,
        query_image: Image.Image,
        query_text: str,
        corpus_embeddings: torch.Tensor,
        corpus_paths: List[str],
        top_k: int = 5,
        top_n_coarse: int = 10,
        max_iterations: int = 2,
        reward_threshold: float = 0.95,
        hybrid_alpha: float = 0.55,
        use_reasoning: bool = True,
        candidate_pool_size: int = 50,
        enable_tournament: bool = True,
    ) -> List[Tuple[str, float, str]]:
        """
        Full Two-Stage Adaptive Retrieval Loop with Cascade Candidate Generation and Memory Bank.

        0. Reason: (Optional, if use_reasoning=True) LLaVA simulates the target video scene
           from [query_image + query_text].
        1. Cascade Candidate Generation:
           - Fast Broad Pool: Scans entire corpus to fetch Top-M (candidate_pool_size=50) candidates in 0.005s.
           - Dual-View Alignment: Balances composed query with target description text.
           - Fast Intent-Preserving Pre-rank: Isolates Top-N optimal candidates.
        2. Agentic Reranker (LLaVA): Top-K ranked results with RRF + Pairwise Tournament.
        3. Critic Agent: Evaluates Top-1 result (instant tensor cache reuse).
        4. If score >= reward_threshold or decisive margin: Early stop (Confirmed High-Quality Hit).
        5. Preserves the clean iteration 0 results to safeguard Recall@5 and Recall@10,
           only superseding if a retry iteration achieves a definitively superior score (>= reward_threshold).
        """
        # Per-query score and tensor caches:
        _score_cache: Dict[str, Tuple[float, str]] = {}
        _tensor_cache: Dict[str, List[torch.Tensor]] = {}
        memory = ToolMemoryBank(
            max_iterations=max_iterations,
            reward_threshold=reward_threshold,
            initial_alpha=0.5,
        )

        # Stage 0: Reason-then-Retrieve visual simulation
        reasoned_desc = None
        if use_reasoning:
            print("\n[ReAgentV] --- Reason-then-Retrieve: Simulating Target Video State ---")
            reasoned_desc = self.simulate_target_description(query_image, query_text)

        best_results = []
        best_reward = -1.0
        best_top1_score = -1.0

        for iteration in range(max_iterations):
            print(f"\n[ReAgentV] === Iteration {iteration + 1}/{max_iterations} ===")

            # Get adaptive strategy from Memory Bank
            strategy = memory.get_current_strategy()
            alpha          = strategy["alpha"]
            exclude_paths  = strategy["exclude_videos"]
            hint           = strategy["query_expansion_hint"]
            force_ocr      = strategy["force_ocr"]
            force_det      = strategy["force_det"]
            failure_feedback = strategy.get("failure_feedback", "")
            current_hybrid_alpha = strategy.get("suggested_hybrid_alpha", hybrid_alpha)

            # In retry iterations: Feedback-Guided Query Refinement (Explicit Negative Guidance)
            expanded_text = query_text
            if iteration > 0:
                try:
                    if failure_feedback:
                        expansion_prompt = covr_feedback_refinement_template.format(
                            edit_prompt=query_text,
                            failure_feedback=failure_feedback[:150],
                        )
                    else:
                        expansion_prompt = covr_query_expansion_template.format(
                            edit_prompt=query_text
                        )
                    raw_hint = llava_inference(expansion_prompt, None, max_new_tokens=64).strip()
                    if "assistant\n" in raw_hint:
                        raw_hint = raw_hint.split("assistant\n")[-1].strip()
                    hint_clean = raw_hint.replace("\n", " ").strip().strip('"\'')
                    if hint_clean:
                        hint = hint_clean[:100].strip()
                        print(f"[ReAgentV] Query refinement hint (feedback-guided): '{hint}'")
                except Exception as e:
                    print(f"[ReAgentV] Query refinement fallback: {e}")

            tools_used = ["CLIP_coarse"]
            if use_reasoning:
                tools_used.append("LLaVA_reason")
            if enable_tournament:
                tools_used.append("Pairwise_Tournament")
            if force_det:
                tools_used.append("DET")
            if force_ocr:
                tools_used.append("OCR")

            # Stage 1: Cascade Candidate Generation (Pool Top-M -> Pre-rank Top-N)
            coarse_results = self.coarse_search(
                query_image, query_text,
                corpus_embeddings, corpus_paths,
                top_n=top_n_coarse,
                alpha=alpha,
                query_expansion_hint=hint,
                exclude_paths=exclude_paths,
                reasoned_description=reasoned_desc, # Restored: drives target scene alignment for replacement queries (goats, lit tree, Mars, moose)
                candidate_pool_size=candidate_pool_size,
            )

            # Stage 2: Agentic Reranker (with RRF + Pairwise Tournament + score cache)
            reranked = self.agentic_rerank(
                query_image, query_text, coarse_results,
                top_k=top_k, hybrid_alpha=current_hybrid_alpha,
                score_cache=_score_cache,
                tensor_cache=_tensor_cache,
                enable_tournament=enable_tournament,
                target_sim=reasoned_desc,
                exclude_paths=exclude_paths,
            )

            if not reranked:
                print("[ReAgentV] No valid reranked results — stopping early.")
                break

            top1_path = reranked[0][0]

            # Stage 3: Critic Agent evaluates current Top-1 (reuses prefetched tensor)
            scalar_reward, critic_raw = self.covr_critic_evaluate(
                query_image, query_text, top1_path,
                tensor_cache=_tensor_cache,
            )

            candidate_paths = [r[0] for r in reranked]

            # Module 3 Safe Blacklist Guard:
            # Protect candidates confirmed as MATCH (rel >= 0.70) or belonging to original CLIP Top-3
            safe_candidates = {
                p for p, (rel, v) in _score_cache.items()
                if v == "MATCH" or rel >= 0.70
            }
            if coarse_results:
                for cr in coarse_results[:3]:
                    safe_candidates.add(cr[0])

            memory.record_step(
                iteration=iteration,
                tools_used=tools_used,
                query_prompt_used=expanded_text,
                top_candidates=candidate_paths,
                critic_raw=critic_raw,
                scalar_reward=scalar_reward,
                safe_candidates=safe_candidates,
            )

            # ── Hướng 2: Bỏ rào cản cứng vô lý, cho phép Vòng 2 cập nhật kết quả hợp lý ──
            curr_top1_score = reranked[0][1] if reranked else 0.0
            if not best_results:
                best_results = reranked
                best_reward = scalar_reward
                best_top1_score = curr_top1_score
            else:
                is_improved = False
                update_reason = ""
                if best_reward < 0.60 and scalar_reward >= 0.60:
                    is_improved = True
                    update_reason = f"Rescued low-confidence baseline (reward {best_reward:.3f} -> {scalar_reward:.3f})"
                elif scalar_reward > best_reward + 0.02:
                    is_improved = True
                    update_reason = f"Higher critic reward ({scalar_reward:.3f} > {best_reward:.3f})"
                elif abs(scalar_reward - best_reward) <= 0.05 and curr_top1_score > best_top1_score + 0.02:
                    is_improved = True
                    update_reason = f"Superior ranking confidence ({curr_top1_score:.3f} > {best_top1_score:.3f})"

                if is_improved:
                    best_results = reranked
                    best_reward = scalar_reward
                    best_top1_score = curr_top1_score
                    print(f"[ReAgentV] Updated best results in iteration {iteration + 1}: {update_reason}")
                else:
                    print(f"[ReAgentV] Retained prior best results (reward={best_reward:.3f}, score={best_top1_score:.3f}) over iter {iteration + 1} (reward={scalar_reward:.3f}, score={curr_top1_score:.3f})")

            # ── Hướng 1: Calibrated Early Stopping (Không dừng sớm mù quáng) ──
            # Only stop early if:
            # 1. Critic gives high reward (>= reward_threshold)
            # 2. Top-1 candidate has genuine visual verification (s_edit >= 0.65 from score_cache)
            # 3. Confidence margin between Top-1 and Top-2 is decisive (>= 0.07), avoiding tie-break ambiguities
            top1_rel = _score_cache.get(top1_path, (0.0, ""))[0]
            score_margin = (reranked[0][1] - reranked[1][1]) if len(reranked) >= 2 else 1.0

            can_early_stop = (
                scalar_reward >= reward_threshold and
                top1_rel >= 0.65 and
                score_margin >= 0.07
            )

            if can_early_stop:
                print(f"[ReAgentV] High confidence hit (reward={scalar_reward:.3f} >= {reward_threshold}, rel={top1_rel:.2f}, margin={score_margin:.3f}) — stopping early.")
                break
            elif scalar_reward >= reward_threshold:
                print(f"[ReAgentV] Critic reward high ({scalar_reward:.3f}) but top candidates in close competition (margin={score_margin:.3f} < 0.07, rel={top1_rel:.2f}) — continuing refinement to resolve ambiguity.")

            # Identical ranking early exit: if retry iteration yields the exact same Top-3 candidates,
            # further iterations will just produce duplicate cache hits without improving accuracy
            if iteration >= 1 and candidate_paths[:3] == memory.history[iteration - 1].top_candidates[:3]:
                print(f"[ReAgentV] Converged on stable Top candidates — stopping early.")
                break

            # Stagnation early exit: if retry iteration yields no improvement over baseline, stop early
            if iteration >= 1 and scalar_reward <= best_reward and scalar_reward < 0.50:
                print(f"[ReAgentV] Stagnation detected (reward {scalar_reward:.3f} <= baseline {best_reward:.3f}) — stopping early to save compute.")
                break

            if not memory.should_continue(scalar_reward, iteration):
                print(f"[ReAgentV] Stopping: max iterations reached.")
                break

        print(memory.summary())
        return best_results if best_results else reranked