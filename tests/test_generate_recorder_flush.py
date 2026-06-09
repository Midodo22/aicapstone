import ast
from pathlib import Path


SOURCE = Path("scripts/datagen/generate.py")


def test_episode_reset_flushes_recorder_before_stop_check():
    tree = ast.parse(SOURCE.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_on_episode_done"
    )

    reset_line = next(
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "env"
        and node.func.attr == "reset"
    )
    stop_check_line = next(
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_should_stop_generating"
    )

    assert reset_line < stop_check_line
