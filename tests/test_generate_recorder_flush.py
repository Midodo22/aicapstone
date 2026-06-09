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


def test_resume_count_checks_huggingface_lerobot_cache():
    source = SOURCE.read_text()

    assert 'os.environ.get("HF_LEROBOT_HOME"' in source
    assert 'os.path.join(hf_lerobot_home, args_cli.lerobot_dataset_repo_id)' in source


def test_resume_validates_episode_metadata_before_environment_creation():
    tree = ast.parse(SOURCE.read_text())
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    validate_line = next(
        node.lineno
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_validate_lerobot_resume_dataset"
    )
    gym_make_line = next(
        node.lineno
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "gym"
        and node.func.attr == "make"
    )

    assert validate_line < gym_make_line
