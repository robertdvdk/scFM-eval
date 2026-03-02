#!/bin/bash
CONTAINER=$(docker ps --filter "label=devcontainer.local_folder=/local/home/rvander/scFM-eval" --format "{{.Names}}")

# Forward WANDB_API_KEY from host .netrc if not already set
if [ -z "$WANDB_API_KEY" ] && [ -f "$HOME/.netrc" ]; then
    WANDB_API_KEY=$(awk '/machine api.wandb.ai/{found=1} found && /password/{print $2; exit}' "$HOME/.netrc")
fi

ENV_FLAGS=()
if [ -n "$WANDB_API_KEY" ]; then
    ENV_FLAGS+=(-e "WANDB_API_KEY=$WANDB_API_KEY")
fi

docker exec -u appuser "${ENV_FLAGS[@]}" "$CONTAINER" bash -c "cd /app && $*"
