#!/usr/bin/env bash
set -euo pipefail

MODELS='task.foundation_models=[cancerfoundation,cancerfoundation_2,scgpt_human,scgpt_pancancer]'

# dat_weight=0.0 (no domain-adaptive training)
python src/main.py task=finetuned_batch_integration "$MODELS" task.dataset_name=kim_lung   task.finetune.dat_weight=5.0
python src/main.py task=finetuned_batch_integration "$MODELS" task.dataset_name=neftel_ss2 task.finetune.dat_weight=5.0
python src/main.py task=finetuned_batch_integration "$MODELS" task.dataset_name=ji_skin    task.finetune.dat_weight=5.0

# dat_weight=10.0
python src/main.py task=finetuned_batch_integration "$MODELS" task.dataset_name=kim_lung   task.finetune.dat_weight=10.0
python src/main.py task=finetuned_batch_integration "$MODELS" task.dataset_name=neftel_ss2 task.finetune.dat_weight=10.0
python src/main.py task=finetuned_batch_integration "$MODELS" task.dataset_name=ji_skin    task.finetune.dat_weight=10.0
