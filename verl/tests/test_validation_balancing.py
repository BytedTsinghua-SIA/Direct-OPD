import ast
from pathlib import Path


def _load_helper():
    source_path = Path(__file__).parents[1] / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    module_ast = ast.parse(source_path.read_text())
    helper_nodes = [
        node
        for node in module_ast.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_per_prompt_balanced_indices"
    ]
    assert helper_nodes, "_build_per_prompt_balanced_indices is missing"

    module = ast.Module(body=helper_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["_build_per_prompt_balanced_indices"]


def test_per_prompt_validation_balancing_spreads_repeats_evenly_across_dp_ranks():
    build_indices = _load_helper()

    indices = build_indices(num_prompts=4, repeat_times=8, dp_size=4)

    assert indices == [
        0,
        1,
        8,
        9,
        16,
        17,
        24,
        25,
        2,
        3,
        10,
        11,
        18,
        19,
        26,
        27,
        4,
        5,
        12,
        13,
        20,
        21,
        28,
        29,
        6,
        7,
        14,
        15,
        22,
        23,
        30,
        31,
    ]


def test_per_prompt_validation_balancing_skips_when_repeats_do_not_divide_dp_size():
    build_indices = _load_helper()

    assert build_indices(num_prompts=4, repeat_times=6, dp_size=4) is None
