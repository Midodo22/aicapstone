#!/usr/bin/env python3
"""Validate and package a LeRobot checkpoint for submission."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path


REQUIRED_FILES = ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json")


def resolve_pretrained_model(source: Path) -> Path:
    """Resolve a run directory, checkpoint directory, or pretrained_model directory."""
    candidates = [
        source,
        source / "pretrained_model",
        source / "checkpoints" / "last" / "pretrained_model",
    ]
    checkpoints_dir = source / "checkpoints"
    if checkpoints_dir.is_dir():
        numbered = sorted(
            (path for path in checkpoints_dir.iterdir() if path.is_dir() and path.name.isdigit()),
            key=lambda path: int(path.name),
            reverse=True,
        )
        candidates.extend(path / "pretrained_model" for path in numbered)

    for candidate in candidates:
        if all((candidate / filename).is_file() for filename in REQUIRED_FILES):
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not find a complete pretrained_model under {source}. "
        f"Required files: {', '.join(REQUIRED_FILES)}"
    )


def validate_pretrained_model(model_dir: Path) -> str:
    """Validate the files needed by LeRobot inference and return the policy type."""
    missing = [filename for filename in REQUIRED_FILES if not (model_dir / filename).is_file()]
    if missing:
        raise ValueError(f"{model_dir} is missing required files: {missing}")

    preprocessor_states = list(model_dir.glob("policy_preprocessor*.safetensors"))
    postprocessor_states = list(model_dir.glob("policy_postprocessor*.safetensors"))
    if not preprocessor_states or not postprocessor_states:
        raise ValueError(
            f"{model_dir} must include preprocessor and postprocessor normalization state files. "
            "Submitting weights without these files changes observation/action scaling."
        )

    with (model_dir / "config.json").open() as file:
        config = json.load(file)
    policy_type = config.get("type")
    if policy_type not in {"act", "diffusion"}:
        raise ValueError(f"Expected an ACT or Diffusion checkpoint, got policy type {policy_type!r}.")
    return policy_type


def make_act_reactive(model_dir: Path, policy_type: str) -> None:
    """Configure a packaged ACT policy to replan every step without action averaging."""
    if policy_type != "act":
        raise ValueError("--reactive-act can only be applied to an ACT checkpoint.")

    config_path = model_dir / "config.json"
    with config_path.open() as file:
        config = json.load(file)
    config["n_action_steps"] = 1
    config["temporal_ensemble_coeff"] = None
    with config_path.open("w") as file:
        json.dump(config, file, indent=2, sort_keys=True)
        file.write("\n")


def package_policy(
    source: Path,
    output_dir: Path,
    zip_path: Path,
    force: bool = False,
    reactive_act: bool = False,
) -> tuple[str, Path]:
    """Copy a validated checkpoint to output_dir and create a zip containing output_dir as its root."""
    model_dir = resolve_pretrained_model(source)
    policy_type = validate_pretrained_model(model_dir)
    if reactive_act and policy_type != "act":
        raise ValueError("--reactive-act can only be applied to an ACT checkpoint.")

    if output_dir.exists():
        if not force:
            raise FileExistsError(f"{output_dir} already exists. Pass --force to replace it.")
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(model_dir, output_dir)
    if reactive_act:
        make_act_reactive(output_dir, policy_type)

    if zip_path.exists():
        if not force:
            raise FileExistsError(f"{zip_path} already exists. Pass --force to replace it.")
        zip_path.unlink()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, Path(output_dir.name) / path.relative_to(output_dir))

    return policy_type, zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Training run, checkpoint, or pretrained_model directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/my_policy"))
    parser.add_argument("--zip-path", type=Path, default=Path("checkpoints/my_policy.zip"))
    parser.add_argument("--force", action="store_true", help="Replace an existing output directory and zip.")
    parser.add_argument(
        "--reactive-act",
        action="store_true",
        help=(
            "Package ACT with n_action_steps=1 and temporal ensembling disabled. "
            "This prevents delayed gripper transitions and makes the checkpoint replan every step."
        ),
    )
    args = parser.parse_args()

    policy_type, zip_path = package_policy(
        args.source,
        args.output_dir,
        args.zip_path,
        args.force,
        args.reactive_act,
    )
    print(f"Packaged {policy_type} policy: {zip_path}")


if __name__ == "__main__":
    main()
