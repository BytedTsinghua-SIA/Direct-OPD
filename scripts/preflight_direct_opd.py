#!/usr/bin/env python3
"""Preflight checks for the 2026-07-02 online Delta-OPD runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoTokenizer


DEFAULT_REPORT_ROOT = Path("/sia-thu/chihaohan/direct-opd-repro/reports")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--actor-model", required=True, type=Path)
    parser.add_argument("--reward-model", required=True, type=Path)
    parser.add_argument("--teacher-ref-model", required=True, type=Path)
    parser.add_argument("--train-dataset", required=True, type=Path)
    parser.add_argument("--eval-dataset", action="append", default=[], type=Path)
    parser.add_argument("--sample-rows", type=int, default=32)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument(
        "--allow-tokenizer-mismatch",
        action="store_true",
        default=os.environ.get("ALLOW_TOKENIZER_MISMATCH", "0") == "1",
    )
    return parser.parse_args()


def check_path(path: Path, label: str) -> dict[str, Any]:
    exists = path.exists()
    return {"label": label, "path": str(path), "exists": exists, "is_dir": path.is_dir() if exists else False}


def load_tokenizer(path: Path) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    return {
        "path": str(path),
        "class": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "has_chat_template": bool(getattr(tokenizer, "chat_template", None)),
    }


def compare_tokenizers(tokenizers: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    base = tokenizers[0]
    comparable_fields = ("vocab_size", "eos_token_id", "bos_token_id")
    for other in tokenizers[1:]:
        for field in comparable_fields:
            if base[field] != other[field]:
                errors.append(
                    f"Tokenizer mismatch for {field}: actor={base[field]} vs {other['path']}={other[field]}"
                )
    if not base["has_chat_template"]:
        errors.append(f"Actor tokenizer has no chat_template: {base['path']}")
    if base["pad_token_id"] is None:
        errors.append(f"Actor tokenizer pad_token_id is None: {base['path']}")
    return errors


def inspect_dataset(path: Path, sample_rows: int) -> dict[str, Any]:
    table = pq.read_table(path)
    rows = table.slice(0, min(sample_rows, table.num_rows)).to_pylist()
    missing_prompt = 0
    missing_ground_truth = 0
    for row in rows:
        if not row.get("prompt"):
            missing_prompt += 1
        reward_model = row.get("reward_model") or {}
        if not reward_model.get("ground_truth"):
            missing_ground_truth += 1
    return {
        "path": str(path),
        "rows": table.num_rows,
        "columns": table.schema.names,
        "sample_rows": len(rows),
        "missing_prompt_in_sample": missing_prompt,
        "missing_ground_truth_in_sample": missing_ground_truth,
    }


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    path_checks = [
        check_path(args.actor_model, "actor_model"),
        check_path(args.reward_model, "reward_model"),
        check_path(args.teacher_ref_model, "teacher_ref_model"),
        check_path(args.train_dataset, "train_dataset"),
    ]
    path_checks.extend(check_path(path, f"eval_dataset_{idx}") for idx, path in enumerate(args.eval_dataset))
    errors.extend(f"Missing {entry['label']}: {entry['path']}" for entry in path_checks if not entry["exists"])

    tokenizers: list[dict[str, Any]] = []
    if not errors:
        for path in (args.actor_model, args.reward_model, args.teacher_ref_model):
            tokenizers.append(load_tokenizer(path))
        tokenizer_errors = compare_tokenizers(tokenizers)
        if tokenizer_errors and not args.allow_tokenizer_mismatch:
            errors.extend(tokenizer_errors)

    dataset_reports: list[dict[str, Any]] = []
    if args.train_dataset.exists():
        dataset_reports.append(inspect_dataset(args.train_dataset, args.sample_rows))
        train_report = dataset_reports[-1]
        if train_report["rows"] <= 0:
            errors.append(f"Train dataset is empty: {args.train_dataset}")
        if train_report["missing_prompt_in_sample"] > 0:
            errors.append(f"Train dataset sample has missing prompt rows: {args.train_dataset}")
        if train_report["missing_ground_truth_in_sample"] > 0:
            errors.append(f"Train dataset sample has missing ground_truth rows: {args.train_dataset}")
    for path in args.eval_dataset:
        if path.exists():
            dataset_reports.append(inspect_dataset(path, args.sample_rows))

    report = {
        "profile": args.profile,
        "ok": not errors,
        "errors": errors,
        "path_checks": path_checks,
        "tokenizers": tokenizers,
        "allow_tokenizer_mismatch": args.allow_tokenizer_mismatch,
        "datasets": dataset_reports,
    }
    args.report_root.mkdir(parents=True, exist_ok=True)
    report_path = args.report_root / f"preflight_{args.profile}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
