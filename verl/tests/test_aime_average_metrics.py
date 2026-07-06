import ast
from pathlib import Path


def _load_helper():
    source_path = Path(__file__).parents[1] / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    module_ast = ast.parse(source_path.read_text())
    helper_nodes = [
        node
        for node in module_ast.body
        if isinstance(node, ast.FunctionDef) and node.name == "_add_aime24_aime25_average_metric"
    ]
    assert helper_nodes, "_add_aime24_aime25_average_metric is missing"

    module = ast.Module(body=helper_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["_add_aime24_aime25_average_metric"]


def test_add_aime24_aime25_average_metric_logs_ave16():
    add_average = _load_helper()
    metrics = {
        "val-core/aime24/acc/mean@16": 0.5,
        "val-core/aime25/acc/mean@16": 0.25,
    }

    add_average(metrics)

    assert metrics["val-core/aime24_aime25/acc/ave@16"] == 0.375


def test_add_aime24_aime25_average_metric_accepts_once_suffixes():
    add_average = _load_helper()
    metrics = {
        "val-core/aime24_once/acc/mean@16": 0.75,
        "val-core/aime25_once/acc/mean@16": 0.25,
    }

    add_average(metrics)

    assert metrics["val-core/aime24_aime25/acc/ave@16"] == 0.5


def test_add_aime24_aime25_average_metric_logs_three_way_ave32():
    add_average = _load_helper()
    metrics = {
        "val-core/aime24_once/acc/mean@32": 0.25,
        "val-core/aime25_once/acc/mean@32": 0.5,
        "val-core/HMMT_Feb_2025/acc/mean@32": 1.0,
    }

    add_average(metrics)

    assert metrics["val-core/aime24_aime25_hmmt_feb/acc/ave@32"] == (0.25 + 0.5 + 1.0) / 3
