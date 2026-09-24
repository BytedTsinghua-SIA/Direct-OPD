"""Privileged KL anchor: score student responses with the frozen reference conditioned on (x, z).

Only numpy/torch are imported at module level so data-prep scripts can load the prompt templates without verl.
"""

import numpy as np
import torch

PRIVILEGED_REF_MODES = ("none", "answer", "solution")

_PRIVILEGED_TEMPLATES = {
    "solution": (
        "\n\nHere is a reference solution:\n{z}\n\n"
        "After understanding the reference solution, please try to solve this problem using your own approach below."
    ),
    "answer": "\n\nThe final answer to this problem is: {z}\n\nPlease solve this problem using your own approach below.",
}


def build_privileged_messages(raw_prompt, z: str, mode: str) -> list[dict]:
    """Append the privileged block for ``mode`` to the last user turn of ``raw_prompt``."""
    if mode not in _PRIVILEGED_TEMPLATES:
        raise ValueError(f"privileged mode must be one of {sorted(_PRIVILEGED_TEMPLATES)}, got {mode!r}")
    if not isinstance(z, str) or not z.strip():
        raise ValueError(f"privileged mode {mode!r} requires a non-empty string, got {z!r}")
    messages = [dict(message) for message in raw_prompt]
    user_indices = [i for i, message in enumerate(messages) if message.get("role") == "user"]
    if not user_indices:
        raise ValueError("raw_prompt has no user message to attach privileged information to")
    last_user = messages[user_indices[-1]]
    last_user["content"] = last_user["content"] + _PRIVILEGED_TEMPLATES[mode].format(z=z.strip())
    return messages


def assemble_ref_inputs(
    prompt_ids_list: list[list[int]],
    responses: torch.Tensor,
    response_attention_mask: torch.Tensor,
    pad_token_id: int,
    max_prompt_length: int,
) -> dict[str, torch.Tensor]:
    """Left-pad per-row prompts to a common width and append the unchanged response tokens.

    The actor slices log-probs from the last ``responses.size(-1)`` positions, so the response block must be the
    exact rollout tokens and mask; only the prompt block may differ from the rollout.
    """
    if len(prompt_ids_list) != responses.size(0):
        raise ValueError(f"got {len(prompt_ids_list)} prompts for {responses.size(0)} responses")
    if response_attention_mask.shape != responses.shape:
        raise ValueError("response_attention_mask must have the same shape as responses")
    prompt_width = max(len(ids) for ids in prompt_ids_list)
    if prompt_width > max_prompt_length:
        raise ValueError(f"privileged prompt has {prompt_width} tokens, above max {max_prompt_length}")

    batch_size = responses.size(0)
    prompt_ids = torch.full((batch_size, prompt_width), pad_token_id, dtype=responses.dtype)
    prompt_mask = torch.zeros((batch_size, prompt_width), dtype=response_attention_mask.dtype)
    for row, ids in enumerate(prompt_ids_list):
        if len(ids) == 0:
            raise ValueError(f"row {row}: empty privileged prompt")
        prompt_ids[row, prompt_width - len(ids) :] = torch.tensor(ids, dtype=responses.dtype)
        prompt_mask[row, prompt_width - len(ids) :] = 1

    input_ids = torch.cat([prompt_ids, responses.cpu()], dim=-1)
    attention_mask = torch.cat([prompt_mask, response_attention_mask.cpu()], dim=-1)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        # same as verl.utils.model.compute_position_id_with_mask
        "position_ids": torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0, max=None),
        "responses": responses.cpu(),
    }


def build_privileged_ref_batch(
    batch: "DataProto",
    tokenizer,
    mode: str,
    max_prompt_length: int,
    chat_template_kwargs: dict | None = None,
) -> tuple["DataProto", dict[str, float]]:
    """Build the ref-only batch whose prompts carry privileged information ``extra_info[mode]``."""
    from verl import DataProto

    if mode not in _PRIVILEGED_TEMPLATES:
        raise ValueError(f"privileged mode must be one of {sorted(_PRIVILEGED_TEMPLATES)}, got {mode!r}")
    if "raw_prompt" not in batch.non_tensor_batch:
        raise ValueError("privileged ref requires data.return_raw_chat=True so raw_prompt reaches the trainer")
    if "extra_info" not in batch.non_tensor_batch:
        raise ValueError(f"privileged ref requires extra_info[{mode!r}] in the training data")

    chat_template_kwargs = dict(chat_template_kwargs or {})
    prompt_ids_list = []
    for row, (raw_prompt, extra_info) in enumerate(
        zip(batch.non_tensor_batch["raw_prompt"], batch.non_tensor_batch["extra_info"], strict=True)
    ):
        if not isinstance(extra_info, dict) or mode not in extra_info:
            raise ValueError(f"row {row}: extra_info has no {mode!r} field")
        messages = build_privileged_messages(list(raw_prompt), extra_info[mode], mode)
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **chat_template_kwargs)
        prompt_ids_list.append(tokenizer(text, add_special_tokens=False)["input_ids"])

    responses = batch.batch["responses"]
    response_length = responses.size(-1)
    tensors = assemble_ref_inputs(
        prompt_ids_list=prompt_ids_list,
        responses=responses,
        response_attention_mask=batch.batch["attention_mask"][:, -response_length:],
        pad_token_id=tokenizer.pad_token_id,
        max_prompt_length=max_prompt_length,
    )
    prompt_lengths = np.array([len(ids) for ids in prompt_ids_list], dtype=np.float64)
    metrics = {
        "privileged_ref/prompt_len_mean": float(prompt_lengths.mean()),
        "privileged_ref/prompt_len_max": float(prompt_lengths.max()),
    }
    return DataProto.from_dict(tensors=tensors), metrics
