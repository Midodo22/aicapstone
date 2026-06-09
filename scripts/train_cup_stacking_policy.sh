#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <act|diffusion> <dataset_repo_id> [output_dir]" >&2
  exit 2
fi

POLICY_TYPE="$1"
DATASET_REPO_ID="$2"
OUTPUT_DIR="${3:-outputs/train/cup_eval_${POLICY_TYPE}_seed42}"

BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
STEPS="${STEPS:-200000}"
SEED="${SEED:-42}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"

if ! command -v lerobot-train >/dev/null 2>&1; then
  echo "lerobot-train not found. Run this on the host after activating the project environment." >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found. Install it before training so TorchCodec can decode dataset videos." >&2
  echo "Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y ffmpeg" >&2
  exit 1
fi

if ! python -c "from torchcodec.decoders import VideoDecoder" >/dev/null 2>&1; then
  echo "TorchCodec cannot load its FFmpeg libraries in the active Python environment." >&2
  echo "Verify ffmpeg is installed and TorchCodec is compatible with the installed PyTorch version." >&2
  exit 1
fi

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Output directory already exists: $OUTPUT_DIR" >&2
  echo "Choose a new output directory or resume the existing run explicitly." >&2
  exit 1
fi

COMMON_ARGS=(
  "--dataset.repo_id=$DATASET_REPO_ID"
  "--dataset.image_transforms.enable=true"
  "--dataset.image_transforms.max_num_transforms=2"
  "--policy.type=$POLICY_TYPE"
  "--policy.device=cuda"
  "--policy.use_amp=true"
  "--policy.push_to_hub=false"
  "--output_dir=$OUTPUT_DIR"
  "--job_name=cup_eval_${POLICY_TYPE}_seed${SEED}"
  "--seed=$SEED"
  "--batch_size=$BATCH_SIZE"
  "--num_workers=$NUM_WORKERS"
  "--steps=$STEPS"
  "--save_checkpoint=true"
  "--save_freq=20000"
  "--log_freq=100"
  "--wandb.enable=$WANDB_ENABLE"
)

case "$POLICY_TYPE" in
  act)
    POLICY_ARGS=(
      "--policy.chunk_size=50"
      "--policy.n_action_steps=1"
      "--policy.temporal_ensemble_coeff=0.01"
    )
    ;;
  diffusion)
    POLICY_ARGS=(
      "--policy.horizon=32"
      "--policy.n_action_steps=8"
      "--policy.drop_n_last_frames=23"
      "--policy.down_dims"
      "256"
      "512"
      "1024"
      "--policy.noise_scheduler_type=DDIM"
      "--policy.num_inference_steps=10"
    )
    ;;
  *)
    echo "Unsupported policy type: $POLICY_TYPE (expected act or diffusion)" >&2
    exit 2
    ;;
esac

echo "Training $POLICY_TYPE on $DATASET_REPO_ID"
echo "Output: $OUTPUT_DIR"
lerobot-train "${COMMON_ARGS[@]}" "${POLICY_ARGS[@]}"

echo
echo "Training complete. Validate this checkpoint with fixed evaluation before packaging:"
echo "  $OUTPUT_DIR/checkpoints/last/pretrained_model"
