import ast
from pathlib import Path


SOURCE = Path("packages/simulator/src/simulator/datagen/state_machine/cup_stacking.py")


def _source_constants():
    tree = ast.parse(SOURCE.read_text())
    module_constants = {}
    class_constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "_GRASP_Z_OFFSET":
                module_constants[target.id] = ast.literal_eval(node.value)
        if isinstance(node, ast.ClassDef) and node.name == "CupStackingStateMachine":
            for child in node.body:
                if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                    if child.target.id == "PHASE_STEPS":
                        class_constants[child.target.id] = ast.literal_eval(child.value)
    return module_constants, class_constants


def _method_references(name):
    tree = ast.parse(SOURCE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return {
                child.id
                for child in ast.walk(node)
                if isinstance(child, ast.Name)
            }
    raise AssertionError(f"Method {name} not found")


def test_compact_trajectory_avoids_teaching_long_pauses():
    _, class_constants = _source_constants()
    phase_steps = class_constants["PHASE_STEPS"]

    assert phase_steps == (150, 80, 35, 100, 130, 70, 40)
    assert sum(phase_steps) == 605
    assert sum(phase_steps) < 1200


def test_gripper_reaches_final_grasp_height_before_closing():
    module_constants, _ = _source_constants()

    assert module_constants["_GRASP_Z_OFFSET"] == 0.08
    assert "_GRASP_Z_OFFSET" in _method_references("_phase_approach_blue")
    assert "_GRASP_Z_OFFSET" in _method_references("_phase_grasp_blue")
