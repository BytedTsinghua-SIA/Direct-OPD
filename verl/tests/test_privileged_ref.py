import importlib.util
from pathlib import Path

import numpy as np
import torch

from verl.trainer.ppo.privileged_ref import (
    assemble_ref_inputs,
    build_privileged_messages,
    build_privileged_ref_batch,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_prepare_script():
    spec = importlib.util.spec_from_file_location(
        "prepare_privileged_math", REPO_ROOT / "scripts" / "prepare_privileged_math.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rollout_layout(prompts, responses, response_mask, pad_token_id, prompt_width):
    """Mimic the vLLM rollout: left-padded prompts, right-padded responses, positions continue past the prompt."""
    batch_size = len(prompts)
    prompt_ids = torch.full((batch_size, prompt_width), pad_token_id)
    prompt_mask = torch.zeros((batch_size, prompt_width), dtype=torch.long)
    for row, ids in enumerate(prompts):
        prompt_ids[row, prompt_width - len(ids) :] = torch.tensor(ids)
        prompt_mask[row, prompt_width - len(ids) :] = 1
    prompt_pos = torch.clip(torch.cumsum(prompt_mask, dim=-1) - 1, min=0)
    response_pos = prompt_pos[:, -1:] + torch.arange(1, responses.size(1) + 1)
    return {
        "input_ids": torch.cat([prompt_ids, responses], dim=-1),
        "attention_mask": torch.cat([prompt_mask, response_mask], dim=-1),
        "position_ids": torch.cat([prompt_pos, response_pos], dim=-1),
    }


def test_assemble_ref_inputs_keeps_response_block_exact():
    responses = torch.tensor([[11, 12, 0], [21, 22, 23]])
    response_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    out = assemble_ref_inputs(
        prompt_ids_list=[[1, 2, 3, 4, 5], [6]],
        responses=responses,
        response_attention_mask=response_mask,
        pad_token_id=0,
        max_prompt_length=8,
    )

    assert out["input_ids"].shape == torch.Size([2, 8])
    assert torch.equal(out["input_ids"][:, -3:], responses)
    assert torch.equal(out["attention_mask"][:, -3:], response_mask)
    assert torch.equal(out["responses"], responses)
    assert torch.equal(out["input_ids"][1, :5], torch.tensor([0, 0, 0, 0, 6]))
    assert torch.equal(out["attention_mask"][1], torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]))
    assert torch.equal(out["position_ids"][1], torch.tensor([0, 0, 0, 0, 0, 1, 2, 3]))


def test_assemble_ref_inputs_matches_rollout_layout_for_the_same_prompt():
    prompts = [[5, 6, 7], [8, 9]]
    responses = torch.tensor([[31, 32, 0, 0], [41, 42, 43, 44]])
    response_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    rollout = _rollout_layout(prompts, responses, response_mask, pad_token_id=0, prompt_width=3)

    out = assemble_ref_inputs(prompts, responses, response_mask, pad_token_id=0, max_prompt_length=3)

    valid = rollout["attention_mask"].bool()
    assert torch.equal(out["input_ids"], rollout["input_ids"])
    assert torch.equal(out["attention_mask"], rollout["attention_mask"])
    assert torch.equal(out["position_ids"][valid], rollout["position_ids"][valid])


def test_assemble_ref_inputs_rejects_prompt_above_max_length():
    try:
        assemble_ref_inputs([[1, 2, 3]], torch.tensor([[4]]), torch.tensor([[1]]), pad_token_id=0, max_prompt_length=2)
    except ValueError as error:
        assert "above max" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_build_privileged_messages_appends_to_last_user_turn_without_mutating_input():
    raw_prompt = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Q"}]

    solution = build_privileged_messages(raw_prompt, "  x = 2 \\boxed{2} ", "solution")
    answer = build_privileged_messages(raw_prompt, "2", "answer")

    assert raw_prompt[1]["content"] == "Q"
    assert solution[0] == raw_prompt[0]
    assert solution[1]["content"].startswith("Q\n\nHere is a reference solution:\nx = 2 \\boxed{2}\n\n")
    assert answer[1]["content"].startswith("Q\n\nThe final answer to this problem is: 2\n\n")


def test_build_privileged_messages_rejects_none_mode_and_empty_z():
    for mode, z in (("none", "2"), ("answer", " ")):
        try:
            build_privileged_messages([{"role": "user", "content": "Q"}], z, mode)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for mode={mode!r} z={z!r}")


class _CharTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"

    def __call__(self, text, add_special_tokens):
        return {"input_ids": [ord(char) for char in text]}


def test_build_privileged_ref_batch_uses_extra_info_and_rollout_responses():
    from verl import DataProto

    responses = torch.tensor([[7, 8, 0], [9, 10, 11]])
    rollout_prompt = torch.full((2, 4), 3)
    batch = DataProto.from_dict(
        tensors={
            "responses": responses,
            "input_ids": torch.cat([rollout_prompt, responses], dim=-1),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1]]),
        },
        non_tensors={
            "raw_prompt": np.array(
                [[{"role": "user", "content": "Q1"}], [{"role": "user", "content": "Q22"}]], dtype=object
            ),
            "extra_info": np.array([{"answer": "1", "solution": "s1"}, {"answer": "2", "solution": "s2"}], dtype=object),
        },
    )
    tokenizer = _CharTokenizer()

    ref_batch, metrics = build_privileged_ref_batch(batch, tokenizer, mode="answer", max_prompt_length=4096)

    expected_first = tokenizer.apply_chat_template(
        build_privileged_messages([{"role": "user", "content": "Q1"}], "1", "answer"), True, False
    )
    first_ids = ref_batch.batch["input_ids"][0][ref_batch.batch["attention_mask"][0].bool()][:-2]
    assert "".join(chr(i) for i in first_ids.tolist()) == expected_first
    assert torch.equal(ref_batch.batch["input_ids"][:, -3:], responses)
    assert torch.equal(ref_batch.batch["attention_mask"][:, -3:], torch.tensor([[1, 1, 0], [1, 1, 1]]))
    assert metrics["privileged_ref/prompt_len_max"] == float(len(expected_first) + 1)


def test_zero_delta_reward_scale_gives_zero_advantages():
    from verl.trainer.ppo.core_algos import compute_token_reward_direct_advantage
    from verl.trainer.ppo.ray_trainer import _scale_delta_opd_rm_scores

    rm_scores = torch.randn(2, 3, 4)
    response_mask = torch.ones(2, 3)

    scaled = _scale_delta_opd_rm_scores(rm_scores, 0.0)
    advantages, _ = compute_token_reward_direct_advantage(token_level_rewards=scaled, response_mask=response_mask)

    assert torch.equal(_scale_delta_opd_rm_scores(rm_scores, 1.0), rm_scores)
    assert torch.count_nonzero(advantages) == 0


def test_prepare_script_boxed_answer_and_consistency_helpers():
    prep = _load_prepare_script()

    assert prep.extract_boxed("so the answer is \\boxed{\\frac{1}{2}}.") == "\\frac{1}{2}"
    assert prep.extract_boxed("no box here") is None
    assert prep.answers_match("\\frac{1}{2}", "\\frac12")
    assert not prep.answers_match("3", "4")
    assert prep.MULTIPLE_CHOICE_RE.match("(C)") and not prep.MULTIPLE_CHOICE_RE.match("C2")
    assert prep.PROOF_RE.search("Prove that n is even") and not prep.PROOF_RE.search("Find n")
    assert prep.is_single_line_answer("\\frac{1}{2}")
    assert not prep.is_single_line_answer("\\begin{pmatrix} 1 & 2 \\end{pmatrix}")
    assert not prep.is_single_line_answer("x = 1 \\\\ y = 2")


def test_prepare_script_privileged_content_matches_trainer_template():
    prep = _load_prepare_script()

    content = prep.privileged_content(" 1+1? ", "2", "answer")
    expected = build_privileged_messages([{"role": "user", "content": prep.student_content("1+1?")}], "2", "answer")

    assert content == expected[0]["content"]
    assert prep.source_quotas(19_200, prep.NUMINAMATH_SOURCE_MIX) == {
        "olympiads": 12_480,
        "aops_forum": 3_840,
        "synthetic_amc": 1_920,
        "math": 960,
    }
