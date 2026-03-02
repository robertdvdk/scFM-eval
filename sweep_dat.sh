#!/bin/bash

trap 'exit 130' INT

BASE_CMD="python src/main.py task=finetuned_batch_integration task.foundation_models=[cancerfoundation] task.dataset_name=kim_lung task.baselines=[]"

# Sweep dat_weight with dat_scale=1.0
for w in 0.0 0.1 1.0 10.0 100.0 1000.0; do
  ./run_in_container.sh $BASE_CMD \
    task.finetune.dat_weight=$w \
    task.finetune.dat_scale=1.0 \
    "hydra.run.dir=outputs/finetuned_batch_integration/kim_lung/test_dat/cancerfoundation_dat_weight${w}_dat_scale1.0"
done

BASE_CMD="python src/main.py task=finetuned_batch_integration task.foundation_models=[cancerfoundation] task.dataset_name=neftel_ss2 task.baselines=[]"

# Sweep dat_weight with dat_scale=1.0
for w in 0.0 0.1 1.0 10.0 100.0 1000.0; do
  ./run_in_container.sh $BASE_CMD \
    task.finetune.dat_weight=$w \
    task.finetune.dat_scale=1.0 \
    "hydra.run.dir=outputs/finetuned_batch_integration/neftel_ss2/test_dat/cancerfoundation_dat_weight${w}_dat_scale1.0"
done

BASE_CMD="python src/main.py task=finetuned_batch_integration task.foundation_models=[cancerfoundation] task.dataset_name=ji_skin task.baselines=[]"

# Sweep dat_weight with dat_scale=1.0
for w in 0.0 0.1 1.0 10.0 100.0 1000.0; do
  ./run_in_container.sh $BASE_CMD \
    task.finetune.dat_weight=$w \
    task.finetune.dat_scale=1.0 \
    "hydra.run.dir=outputs/finetuned_batch_integration/ji_skin/test_dat/cancerfoundation_dat_weight${w}_dat_scale1.0"
done
