import json
import zipfile
from pathlib import Path

import pytest

from scripts.package_policy import package_policy, resolve_pretrained_model


def _write_fake_model(model_dir: Path, policy_type: str = "act") -> None:
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({"type": policy_type}))
    (model_dir / "model.safetensors").write_bytes(b"weights")
    (model_dir / "policy_preprocessor.json").write_text("{}")
    (model_dir / "policy_postprocessor.json").write_text("{}")
    (model_dir / "policy_preprocessor_step_3_normalizer_processor.safetensors").write_bytes(b"input-stats")
    (model_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors").write_bytes(b"output-stats")


def test_resolve_pretrained_model_from_training_run(tmp_path):
    model_dir = tmp_path / "run" / "checkpoints" / "last" / "pretrained_model"
    _write_fake_model(model_dir)

    assert resolve_pretrained_model(tmp_path / "run") == model_dir.resolve()


def test_package_policy_creates_submission_root_and_all_files(tmp_path):
    model_dir = tmp_path / "run" / "checkpoints" / "last" / "pretrained_model"
    _write_fake_model(model_dir)
    output_dir = tmp_path / "checkpoints" / "my_policy"
    zip_path = tmp_path / "checkpoints" / "my_policy.zip"

    policy_type, packaged_zip = package_policy(tmp_path / "run", output_dir, zip_path)

    assert policy_type == "act"
    assert packaged_zip == zip_path
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    assert "my_policy/config.json" in names
    assert "my_policy/model.safetensors" in names
    assert "my_policy/policy_preprocessor_step_3_normalizer_processor.safetensors" in names
    assert "my_policy/policy_postprocessor_step_0_unnormalizer_processor.safetensors" in names


def test_package_policy_rejects_missing_normalization_state(tmp_path):
    model_dir = tmp_path / "pretrained_model"
    _write_fake_model(model_dir)
    (model_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors").unlink()

    with pytest.raises(ValueError, match="normalization state"):
        package_policy(model_dir, tmp_path / "my_policy", tmp_path / "my_policy.zip")
