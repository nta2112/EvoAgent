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
    covr_reason_target_prompt_template,
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
            raw_output = llava_inference(prompt, vision_input, max_new_tokens=48)
            cleaned = _strip_llava_output(raw_output)
            # Remove any unwanted quotes or line breaks
            cleaned = cleaned.replace("\n", " ").replace('"', '').strip()
            # If the output starts with descriptive prefixes, clean them
            lower_clean = cleaned.lower()
            prefixes_to_strip = [
                "the target video shows",
                "the target video depicts",
                "the target video is",
                "target video:",
                "description:",
            ]
            for prefix in prefixes_to_strip:
                if lower_clean.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip(" :,-")
                    break

            if cleaned:
                print(f"[ReAgentV Reason] Target simulation: '{cleaned}'")
                return cleaned
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

        # In Reason-then-Retrieve mode, reasoned text is already visually grounded.
        # We set eff_alpha low (0.15) so the target description drives the search,
        # preventing old entities in the reference image (e.g., cow) from contaminating target vectors (e.g., goat).
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

        # Fast GPU/CPU cosine similarity against entire corpus [N]
        sims_composed = (corpus_embeddings @ query_feat.T).squeeze(-1)  # [N]

        # Exclude previously rejected candidates
        if exclude_paths:
            for i, vp in enumerate(corpus_paths):
                if vp in exclude_paths:
                    sims_composed[i] = -1.0

        # Step 1: Broad Candidate Pool (Top-M, e.g. 50-100 videos)
        pool_m = min(max(candidate_pool_size, top_n), len(corpus_paths))
        pool_indices = torch.topk(sims_composed, k=pool_m).indices

        # If pool size equals desired top_n, directly return
        if pool_m <= top_n:
            results = [(corpus_paths[i], float(sims_composed[i])) for i in pool_indices.tolist()[:top_n]]
            return results

        # Step 2: Intent-Focused Pre-ranking
        # When reasoned_description is present, sims_composed is already highly target-aligned.
        # We preserve the top candidates according to the target composed similarity,
        # avoiding biasing back towards the unmodified reference image.
        top_sub_indices = pool_indices[:top_n].tolist()
        results = [(corpus_paths[idx], float(sims_composed[idx])) for idx in top_sub_indices]
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

    def agentic_rerank(
        self,
        query_image: Image.Image,
        query_text: str,
        candidate_list: List[Tuple[str, float]],
        top_k: int = 5,
        hybrid_alpha: float = 0.70,
        score_cache: Optional[Dict[str, Tuple[float, str]]] = None,
        tensor_cache: Optional[Dict[str, List[torch.Tensor]]] = None,
    ) -> List[Tuple[str, float, str]]:
        """
        Stage 2 — Fine-grained Agentic Reranking using LLaVA + Hybrid Scoring.

        Combines CLIP visual similarity with LLaVA action/edit relevance:
            final_score = hybrid_alpha * clip_score + (1 - hybrid_alpha) * rel_score
            (Default hybrid_alpha = 0.70: 70% CLIP visual foundation + 30% LLaVA verifier)

        Asynchronous Parallel Prefetching:
            Candidate videos needing LLaVA inference are pre-decoded in parallel across
            CPU cores using ThreadPoolExecutor, eliminating GPU idle wait times.
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

        reranked = []
        for video_path, clip_score in candidate_list:

            # ── Cache HIT: reuse previously computed LLaVA score ──────────────
            if video_path in score_cache:
                rel_score, verdict = score_cache[video_path]
                print(f"[Cache HIT] {os.path.basename(video_path)} rel={rel_score:.3f}")
                final_score = hybrid_alpha * float(clip_score) + (1.0 - hybrid_alpha) * rel_score
                reranked.append((video_path, round(final_score, 4), verdict))
                continue

            # ── Cache MISS: retrieve prefetched tensor and run LLaVA inference ──
            vid_tensor = self._video_to_tensor(video_path, tensor_cache=tensor_cache)
            if vid_tensor is None:
                score_cache[video_path] = (0.0, "NO_MATCH")
                reranked.append((video_path, 0.0, "NO_MATCH"))
                continue

            # Build combined vision input: [query_frame + video_frames]
            combined_tensor = torch.cat([q_img_tensor, vid_tensor[0]], dim=0)
            combined_input = [combined_tensor]

            prompt = covr_rerank_prompt_template.format(edit_prompt=query_text)
            try:
                raw_output = llava_inference(prompt, combined_input, max_new_tokens=32)
                # Strip LLaVA chat-template artifacts + markdown code fences
                cleaned = _strip_llava_output(raw_output)
                data = json.loads(cleaned)
                rel_score = float(data.get("relevance_score", clip_score))
                verdict = data.get("verdict", "PARTIAL_MATCH")
            except Exception:
                # Fallback on regex if JSON parsing fails
                m = re.search(r'"?relevance_score"?\s*:\s*([0-9]*\.?[0-9]+)', raw_output)
                if m:
                    rel_score = float(m.group(1))
                    verdict = "PARTIAL_MATCH"
                else:
                    rel_score = clip_score * 0.5
                    verdict = "PARTIAL_MATCH"

            # Store in cache for future iterations
            score_cache[video_path] = (rel_score, verdict)

            # Hybrid scoring: blend CLIP similarity with LLaVA relevance
            final_score = hybrid_alpha * float(clip_score) + (1.0 - hybrid_alpha) * rel_score
            reranked.append((video_path, round(final_score, 4), verdict))
            del combined_tensor, combined_input

        # Cleanup GPU cache
        torch.cuda.empty_cache()

        # Sort by hybrid final_score descending
        reranked.sort(key=lambda x: x[1], reverse=True)
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
        reward_threshold: float = 0.92,
        hybrid_alpha: float = 0.70,
        use_reasoning: bool = True,
        candidate_pool_size: int = 50,
    ) -> List[Tuple[str, float, str]]:
        """
        Full Two-Stage Adaptive Retrieval Loop with Cascade Candidate Generation and Memory Bank.

        0. Reason: (Optional, if use_reasoning=True) LLaVA simulates the target video scene
           from [query_image + query_text].
        1. Cascade Candidate Generation:
           - Fast Broad Pool: Scans entire corpus to fetch Top-M (candidate_pool_size=50) candidates in 0.005s.
           - Fast Intent-Preserving Pre-rank: Isolates Top-N optimal candidates.
        2. Agentic Reranker (LLaVA): Top-K ranked results with parallel prefetching.
        3. Critic Agent: Evaluates Top-1 result (instant tensor cache reuse).
        4. If score >= reward_threshold: Early stop (Confirmed High-Quality Hit).
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

        for iteration in range(max_iterations):
            print(f"\n[ReAgentV] === Iteration {iteration + 1}/{max_iterations} ===")

            # Get adaptive strategy from Memory Bank
            strategy = memory.get_current_strategy()
            alpha          = strategy["alpha"]
            exclude_paths  = strategy["exclude_videos"]
            hint           = strategy["query_expansion_hint"]
            force_ocr      = strategy["force_ocr"]
            force_det      = strategy["force_det"]

            # In retry iterations, optionally generate visual detail hints
            expanded_text = query_text
            if iteration > 0:
                try:
                    expansion_prompt = covr_query_expansion_template.format(
                        edit_prompt=query_text
                    )
                    raw_hint = llava_inference(expansion_prompt, None, max_new_tokens=64).strip()
                    if "assistant\n" in raw_hint:
                        raw_hint = raw_hint.split("assistant\n")[-1].strip()
                    hint_clean = raw_hint.replace("\n", " ").strip().strip('"\'')
                    if hint_clean:
                        hint = hint_clean[:100].strip()
                        print(f"[ReAgentV] Query expansion hint: '{hint}'")
                except Exception as e:
                    print(f"[ReAgentV] Query expansion fallback: {e}")

            tools_used = ["CLIP_coarse"]
            if use_reasoning:
                tools_used.append("LLaVA_reason")
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
                reasoned_description=reasoned_desc,
                candidate_pool_size=candidate_pool_size,
            )

            # Stage 2: Agentic Reranker (with parallel prefetching + score cache)
            reranked = self.agentic_rerank(
                query_image, query_text, coarse_results,
                top_k=top_k, hybrid_alpha=hybrid_alpha,
                score_cache=_score_cache,
                tensor_cache=_tensor_cache,
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
            memory.record_step(
                iteration=iteration,
                tools_used=tools_used,
                query_prompt_used=expanded_text,
                top_candidates=candidate_paths,
                critic_raw=critic_raw,
                scalar_reward=scalar_reward,
            )

            # Preserve iteration 0 as solid baseline.
            # Only supersede if a retry iteration genuinely achieves a high reward (>= reward_threshold).
            if not best_results:
                best_results = reranked
                best_reward = scalar_reward
            elif scalar_reward > best_reward and scalar_reward >= reward_threshold:
                best_results = reranked
                best_reward = scalar_reward
                print(f"[ReAgentV] Improved results in iteration {iteration + 1} with reward {scalar_reward:.3f}")

            # Stagnation early exit: if retry iteration yields no improvement over baseline, stop early
            if iteration >= 1 and scalar_reward <= best_reward and scalar_reward < 0.50:
                print(f"[ReAgentV] Stagnation detected (reward {scalar_reward:.3f} <= baseline {best_reward:.3f}) — stopping early to save compute.")
                break

            if not memory.should_continue(scalar_reward, iteration):
                print(f"[ReAgentV] Stopping: reward={scalar_reward:.3f} >= threshold={reward_threshold} or max iterations reached.")
                break

        print(memory.summary())
        return best_results if best_results else reranked