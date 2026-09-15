from ReAgentV_utils.prompt_builder.prompt import (
    tool_retrieval_prompt_template,
    eval_reward_prompt_template,
    conservative_template_str,
    neutral_template_str,
    aggressive_template_str,
    meta_agent_prompt_template,
    critic_template_str
)
import json


from string import Template
from ReAgentV_utils.model_inference.model_inference import llava_inference

def evaluate_answer(question: str, answer: str, context_info: dict, video) -> list[str]:
    import re
    critic_prompt = Template(critic_template_str).substitute(
        question=question,
        answer=answer,
        context=json.dumps(context_info, indent=2)
    )

    critique_response = llava_inference(critic_prompt, video)
    if not critique_response:
        return []

    # 1. Thử parse trực tiếp JSON
    try:
        data = json.loads(critique_response.strip())
        if isinstance(data, list):
            return [str(q).strip() for q in data if q and str(q).strip()]
        if isinstance(data, str) and data.strip():
            return [data.strip()]
    except Exception:
        pass

    # 2. Xóa markdown codeblock ```json ... ``` nếu có
    try:
        cleaned = re.sub(r"^```(?:json)?\s*", "", critique_response.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [str(q).strip() for q in data if q and str(q).strip()]
    except Exception:
        pass

    # 3. Regex tìm mảng JSON trong văn bản: [ ... ]
    try:
        match = re.search(r"\[\s*(.*?)\s*\]", critique_response, re.DOTALL)
        if match:
            arr_str = match.group(0)
            data = json.loads(arr_str)
            if isinstance(data, list):
                return [str(q).strip() for q in data if q and str(q).strip()]
    except Exception:
        pass

    # 4. Fallback tìm các câu hỏi riêng biệt có dấu hỏi (?)
    questions = re.findall(r"(?:^|\n)\s*(?:\d+[\.\)]\s*|-\s*)?([A-Z][^?\n]+\?)", critique_response)
    if questions:
        return [q.strip() for q in questions if len(q.strip()) > 5]

    return []