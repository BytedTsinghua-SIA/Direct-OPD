#!/usr/bin/env python3
"""Convert the Skywork-OR1 math split to the Direct-OPD training format."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


EXPECTED_ROWS = 105_055
PROMPT_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n"
)
PROMPT_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'

MESSAGE_TYPE = pa.list_(pa.struct([("content", pa.string()), ("role", pa.string())]))
REWARD_MODEL_TYPE = pa.struct([("ground_truth", pa.string()), ("style", pa.string())])
MODEL_DIFFICULTY_TYPE = pa.struct(
    [
        ("DeepSeek-R1-Distill-Qwen-1.5B", pa.int64()),
        ("DeepSeek-R1-Distill-Qwen-32B", pa.int64()),
        ("DeepSeek-R1-Distill-Qwen-7B", pa.int64()),
    ]
)
EXTRA_INFO_TYPE = pa.struct(
    [
        ("index", pa.int64()),
        ("model_difficulty", MODEL_DIFFICULTY_TYPE),
        ("original_data_source", pa.string()),
        ("prompt_style", pa.string()),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Official Skywork math parquet")
    parser.add_argument("--output", required=True, type=Path, help="Converted Direct-OPD parquet")
    parser.add_argument(
        "--verify-against",
        type=Path,
        help="Optionally compare every converted field with an existing parquet",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    return parser.parse_args()


def parse_ground_truth(value: Any, row_index: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"row {row_index}: reward_model.ground_truth must be a JSON string")
    try:
        answers = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"row {row_index}: invalid ground-truth JSON") from error
    if not isinstance(answers, list) or len(answers) != 1 or not isinstance(answers[0], str):
        raise ValueError(f"row {row_index}: expected exactly one string ground truth")
    return answers[0]


def convert_row(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or len(prompt) != 1:
        raise ValueError(f"row {row_index}: expected exactly one prompt message")
    message = prompt[0]
    if message.get("role") != "user" or not isinstance(message.get("content"), str):
        raise ValueError(f"row {row_index}: expected one string-valued user message")

    reward_model = row.get("reward_model")
    extra_info = row.get("extra_info")
    if not isinstance(reward_model, dict) or not isinstance(extra_info, dict):
        raise ValueError(f"row {row_index}: missing reward_model or extra_info")
    if reward_model.get("style") != "rule":
        raise ValueError(f"row {row_index}: expected reward style 'rule'")
    if row.get("ability") != "math":
        raise ValueError(f"row {row_index}: expected ability 'math'")

    data_source = row.get("data_source")
    difficulty = extra_info.get("model_difficulty")
    if not isinstance(data_source, str) or not isinstance(difficulty, dict):
        raise ValueError(f"row {row_index}: missing data source or model difficulty")

    question = message["content"].strip()
    return {
        "data_source": "math_dapo",
        "prompt": [{"content": f"{PROMPT_PREFIX}{question}{PROMPT_SUFFIX}", "role": "user"}],
        "ability": "MATH",
        "reward_model": {
            "ground_truth": parse_ground_truth(reward_model.get("ground_truth"), row_index),
            "style": "rule-lighteval/MATH_v2",
        },
        "extra_info": {
            "index": extra_info.get("index"),
            "model_difficulty": difficulty,
            "original_data_source": data_source,
            "prompt_style": "dapo_original",
        },
    }


def convert_table(source: pa.Table) -> pa.Table:
    if source.num_rows != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS:,} rows, found {source.num_rows:,}")
    required_columns = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
    missing_columns = required_columns.difference(source.column_names)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"input is missing required columns: {missing}")

    rows = [convert_row(row, index) for index, row in enumerate(source.to_pylist())]
    return pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA)


def assert_same_data(expected: pa.Table, actual: pa.Table, label: str) -> None:
    expected = expected.replace_schema_metadata(None).combine_chunks()
    actual = actual.replace_schema_metadata(None).combine_chunks()
    if not expected.schema.equals(actual.schema, check_metadata=False):
        raise ValueError(f"{label}: schema differs\nexpected: {expected.schema}\nactual: {actual.schema}")
    if expected.num_rows != actual.num_rows:
        raise ValueError(f"{label}: row count differs: {expected.num_rows} != {actual.num_rows}")
    for name in expected.column_names:
        if not expected[name].equals(actual[name]):
            raise ValueError(f"{label}: column {name!r} differs")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(table: pa.Table, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        pq.write_table(table, temporary, compression="snappy", row_group_size=table.num_rows)
        assert_same_data(table, pq.read_table(temporary), "written output")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input parquet does not exist: {args.input}")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists (pass --overwrite to replace it): {args.output}")
    if args.verify_against is not None and not args.verify_against.is_file():
        raise SystemExit(f"Verification parquet does not exist: {args.verify_against}")

    converted = convert_table(pq.read_table(args.input))
    write_atomic(converted, args.output)

    print(f"Wrote {converted.num_rows:,} rows to {args.output}")
    print(f"SHA256: {sha256(args.output)}")
    if args.verify_against is not None:
        assert_same_data(converted, pq.read_table(args.verify_against), "verification parquet")
        print(f"Verified all rows and fields against {args.verify_against}")


if __name__ == "__main__":
    main()
