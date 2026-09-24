#!/usr/bin/env python3
"""Build a fixed-size math training set with privileged answers and worked solutions.

Sources:
  openthoughts_opsd  siyanzhao/Openthoughts_math_30k_opsd (problem, solution, Answer, correct)
  numinamath_cot     AI-MO/NuminaMath-CoT (problem, solution, source); answer = last \\boxed{} of the solution

The student prompt matches scripts/prepare_skywork_math.py. ``extra_info.answer`` and ``extra_info.solution`` feed
``algorithm.privileged_ref_mode`` (see verl/verl/trainer/ppo/privileged_ref.py).
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_skywork_math import MESSAGE_TYPE, PROMPT_PREFIX, PROMPT_SUFFIX, REWARD_MODEL_TYPE, sha256, write_atomic  # noqa: E402


def _load_module(name: str, path: Path):
    # Load single files directly so this script does not import the full verl package.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_math_dapo = _load_module("math_dapo", REPO_ROOT / "verl/verl/utils/reward_score/math_dapo.py")
_privileged_ref = _load_module("privileged_ref", REPO_ROOT / "verl/verl/trainer/ppo/privileged_ref.py")
build_privileged_messages = _privileged_ref.build_privileged_messages

# Source mix of the NuminaMath-CoT subset in "Rethinking Privileged Information in OPSD" (arXiv 2608.18271).
NUMINAMATH_SOURCE_MIX = {"olympiads": 0.65, "aops_forum": 0.20, "synthetic_amc": 0.10, "math": 0.05}

PROOF_RE = re.compile(r"\b(?:prove|show that|demonstrate that|verify that)\b", re.I)
MULTIPLE_CHOICE_RE = re.compile(r"^\(?[A-E]\)?$")

EXTRA_INFO_TYPE = pa.struct(
    [
        ("index", pa.int64()),
        ("original_data_source", pa.string()),
        ("prompt_style", pa.string()),
        ("answer", pa.string()),
        ("solution", pa.string()),
    ]
)
OUTPUT_SCHEMA = pa.schema(
    [
        ("data_source", pa.string()),
        ("prompt", MESSAGE_TYPE),
        ("ability", pa.string()),
        ("reward_model", REWARD_MODEL_TYPE),
        ("extra_info", EXTRA_INFO_TYPE),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, choices=["openthoughts_opsd", "numinamath_cot"])
    parser.add_argument("--input", required=True, help="Glob of the source parquet shards")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", default=str(REPO_ROOT / "models/Qwen3-1.7B"), help="Student tokenizer")
    parser.add_argument("--num-rows", type=int, default=19_200, help="300 steps x 64 prompts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-student-tokens", type=int, default=1024)
    parser.add_argument("--max-privileged-tokens", type=int, default=2048)
    parser.add_argument(
        "--eval-glob", default=str(REPO_ROOT / "datasets/eval/*.parquet"), help="Benchmarks to decontaminate against"
    )
    parser.add_argument("--exclude", help="Glob of parquet shards with a 'problem' column whose problems are dropped")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def extract_boxed(text: Optional[str]) -> Optional[str]:
    if not isinstance(text, str):
        return None
    boxed = _math_dapo.last_boxed_only_string(text)
    if boxed is None:
        return None
    try:
        return _math_dapo.remove_boxed(boxed)
    except (AssertionError, IndexError):
        return None


def answers_match(boxed: Optional[str], answer: Optional[str]) -> bool:
    if not isinstance(boxed, str) or not isinstance(answer, str):
        return False
    return _math_dapo.normalize_final_answer(boxed) == _math_dapo.normalize_final_answer(answer)


def is_single_line_answer(answer: str) -> bool:
    # The student must state the answer on one "Answer:" line; aligned/matrix/case answers cannot be graded that way.
    return "\n" not in answer and "\\begin" not in answer and "\\\\" not in answer


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def ngrams(text: str, n: int = 13) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def student_content(problem: str) -> str:
    return f"{PROMPT_PREFIX}{problem.strip()}{PROMPT_SUFFIX}"


def privileged_content(problem: str, z: str, mode: str) -> str:
    messages = build_privileged_messages([{"role": "user", "content": student_content(problem)}], z, mode)
    return messages[-1]["content"]


def read_shards(pattern: str, columns: list[str]) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No parquet files match {pattern}")
    return pd.concat([pd.read_parquet(path, columns=columns) for path in paths], ignore_index=True)


def load_source(source: str, pattern: str) -> pd.DataFrame:
    if source == "openthoughts_opsd":
        df = read_shards(pattern, ["source", "problem", "solution", "Answer", "correct"])
        df = df.rename(columns={"Answer": "answer"})
        df["answer"] = df["answer"].where(df["correct"].fillna(False).astype(bool), None)
    else:
        df = read_shards(pattern, ["source", "problem", "solution"])
        df = df[df["source"].isin(NUMINAMATH_SOURCE_MIX)].reset_index(drop=True)
        df["answer"] = df["solution"].map(extract_boxed)
    df["answer"] = df["answer"].map(lambda value: value.strip() if isinstance(value, str) else None)
    return df[["source", "problem", "solution", "answer"]]


class TokenCounter:
    def __init__(self, tokenizer_path: str):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "@@SPLIT@@"}], add_generation_prompt=True, tokenize=False
        )
        before, after = rendered.split("@@SPLIT@@")
        # The template pieces begin and end at special tokens, so no BPE merge crosses them and the sum is exact.
        self.template_overhead = self.count([before])[0] + self.count([after])[0]

    def count(self, texts: list[str]) -> np.ndarray:
        lengths = []
        for start in range(0, len(texts), 20_000):
            encoded = self.tokenizer(texts[start : start + 20_000], add_special_tokens=False)["input_ids"]
            lengths.extend(len(ids) for ids in encoded)
        return np.array(lengths, dtype=np.int64)

    def prompt_lengths(self, contents: list[str]) -> np.ndarray:
        return self.count(contents) + self.template_overhead


def filter_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    print(f"{args.source}: {len(df):,} rows")

    def keep(label: str, mask) -> None:
        nonlocal df
        before = len(df)
        df = df[np.asarray(mask, dtype=bool)].reset_index(drop=True)
        print(f"  {label:<48s} -{before - len(df):>7,} -> {len(df):>7,}")

    keep("has final answer", df["answer"].map(lambda a: isinstance(a, str) and a != ""))
    df["boxed"] = df["solution"].map(extract_boxed)
    keep("solution has \\boxed{}", df["boxed"].notna())
    keep("boxed(solution) matches answer", [answers_match(b, a) for b, a in zip(df["boxed"], df["answer"])])
    keep("single-line answer", df["answer"].map(is_single_line_answer))
    keep("not a multiple-choice letter", ~df["answer"].str.match(MULTIPLE_CHOICE_RE))
    keep("not a proof problem", ~df["problem"].str.contains(PROOF_RE))
    keep("exactly one \\boxed{} in solution", df["solution"].str.count(r"\\boxed") == 1)
    keep("first copy of each problem", ~df["problem"].map(normalize_text).duplicated())

    eval_problems = [
        message[0]["content"].replace(PROMPT_PREFIX, "").replace(PROMPT_SUFFIX, "")
        for path in sorted(glob.glob(args.eval_glob))
        for message in pd.read_parquet(path, columns=["prompt"])["prompt"]
    ]
    if not eval_problems:
        raise SystemExit(f"No eval parquets match {args.eval_glob}")
    eval_norm = [normalize_text(problem) for problem in eval_problems]
    eval_grams = set().union(*(ngrams(problem) for problem in eval_problems))

    def contaminated(problem: str) -> bool:
        norm = normalize_text(problem)
        if any(len(ev) > 30 and (ev in norm or norm in ev) for ev in eval_norm):
            return True
        return bool(ngrams(problem) & eval_grams)

    keep(f"not in {len(eval_problems)} eval problems", ~df["problem"].map(contaminated))

    counter = TokenCounter(args.tokenizer)
    df["student_len"] = counter.prompt_lengths([student_content(p) for p in df["problem"]])
    df["privileged_len"] = np.maximum(
        counter.prompt_lengths([privileged_content(p, s, "solution") for p, s in zip(df["problem"], df["solution"])]),
        counter.prompt_lengths([privileged_content(p, a, "answer") for p, a in zip(df["problem"], df["answer"])]),
    )
    keep(f"student prompt <= {args.max_student_tokens} tokens", df["student_len"] <= args.max_student_tokens)
    keep(f"privileged prompt <= {args.max_privileged_tokens} tokens", df["privileged_len"] <= args.max_privileged_tokens)

    if args.exclude:
        excluded = set(read_shards(args.exclude, ["problem"])["problem"].map(normalize_text))
        keep("not in --exclude problems", ~df["problem"].map(normalize_text).isin(excluded))
    return df


def source_quotas(num_rows: int, mix: dict[str, float]) -> dict[str, int]:
    quotas = {source: int(num_rows * weight) for source, weight in mix.items()}
    largest = max(mix, key=mix.get)
    quotas[largest] += num_rows - sum(quotas.values())
    return quotas


def sample_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.source == "numinamath_cot":
        parts = []
        for source, quota in source_quotas(args.num_rows, NUMINAMATH_SOURCE_MIX).items():
            pool = df[df["source"] == source]
            if len(pool) < quota:
                raise SystemExit(f"{source}: need {quota:,} rows, only {len(pool):,} survive the filters")
            parts.append(pool.sample(n=quota, random_state=args.seed))
            print(f"  sampled {quota:>6,} / {len(pool):>7,} from {source}")
        sampled = pd.concat(parts)
    else:
        if len(df) < args.num_rows:
            raise SystemExit(f"need {args.num_rows:,} rows, only {len(df):,} survive the filters")
        sampled = df.sample(n=args.num_rows, random_state=args.seed)
    # One fixed order shared by every method (the launch script sets data.shuffle=False).
    return sampled.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)


def convert_row(row: pd.Series, index: int, source: str) -> dict:
    return {
        "data_source": "math_dapo",
        "prompt": [{"content": student_content(row["problem"]), "role": "user"}],
        "ability": "MATH",
        "reward_model": {"ground_truth": row["answer"], "style": "rule-lighteval/MATH_v2"},
        "extra_info": {
            "index": index,
            "original_data_source": f"{source}/{row['source']}",
            "prompt_style": "dapo_original",
            "answer": row["answer"],
            "solution": row["solution"].strip(),
        },
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists (pass --overwrite to replace it): {args.output}")

    df = filter_rows(load_source(args.source, args.input), args)
    sampled = sample_rows(df, args)
    rows = [convert_row(row, index, args.source) for index, row in sampled.iterrows()]
    table = pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA)
    write_atomic(table, args.output)

    print(f"Wrote {table.num_rows:,} rows to {args.output}")
    print(f"SHA256: {sha256(args.output)}")


if __name__ == "__main__":
    main()
