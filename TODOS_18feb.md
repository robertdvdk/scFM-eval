# TODOs 18 Feb 2026

## WassersteinDiscriminator: grad_reverse must be added back

### What happened
The original `WassersteinDiscriminator` in `/CancerFoundation/cancerfoundation/model/module.py` had `grad_reverse(x, scale=1.0)` in its `forward()`. I (Claude) incorrectly advised removing it, claiming it caused a "double negation" with the Wasserstein loss. That analysis was wrong.

### Why the original was correct
`grad_reverse` is applied to the **input tensor** (cell embeddings), not to the discriminator's own weights. During backward:
- **Discriminator weights** are downstream of `grad_reverse` in the computation graph, so they receive **normal** gradients. The critic learns to maximize the Wasserstein gap between batches.
- **Encoder weights** are upstream of `grad_reverse`, so they receive **reversed** gradients. The encoder learns to minimize the gap (make batches indistinguishable).

This is exactly the desired adversarial dynamic with a single optimizer.

### Why the "fix" was broken
The detach-based approach (`discriminator(cell_emb.detach())` for critic + `discriminator(cell_emb)` for encoder with negated loss) produces identical forward values since `.detach()` only strips gradient history. The two loss terms cancel to exactly 0 — both in scalar value AND in discriminator gradients. The discriminator never learns.

### What still needs to change

1. **Add `grad_reverse` back to `WassersteinDiscriminator.forward()`** in `/CancerFoundation/cancerfoundation/model/module.py`:
   ```python
   def forward(self, x: Tensor) -> Tensor:
       x = grad_reverse(x, scale=1.0)  # <-- ADD THIS BACK
       for layer in self._decoder:
           x = layer(x)
       return self.out_layer(x)
   ```
   Problem: `/CancerFoundation` is mounted read-only in the devcontainer (see `.devcontainer/devcontainer.json` line 28). Options:
   - Change the mount to read-write (`readonly` -> remove the flag)
   - Or: define `WassersteinDiscriminator` locally in `finetune_cancerfoundation.py` instead of importing from the read-only package
   - Or: apply `grad_reverse` in the training loop before passing to the discriminator

2. **The training loop in `finetune_cancerfoundation.py` is already reverted** to the simple single-loss form:
   ```python
   batch_pred = discriminator(cell_emb.float())
   dat_loss = wasserstein_condition_loss(batch_pred, batch_labels)
   loss = loss + dat_weight * dat_loss
   ```
   This is correct, assuming `grad_reverse` is restored inside the discriminator.

3. **The `wasserstein_condition_loss` function is correct as-is.** Loss = `-(correct_scores - incorrect_scores).mean()`. Minimizing this pushes the critic to separate batches; reversed gradients push the encoder to merge them.

## Other changes made today (these are done)

- **wandb config key fix**: `finetune_cancerfoundation.py` was reading `ft_cfg.get("use_wandb")` but config uses `wandb`. Fixed to `ft_cfg.get("wandb")`.
- **Graceful wandb fallback**: try/except around `import wandb` so it doesn't crash if not installed.
- **wandb installed** in both `envs/cancerfoundation/.venv` and `/opt/venv`.
- **`run_in_container.sh`**: now forwards `WANDB_API_KEY` from host `~/.netrc` into the container.
- **UMAP + benchmark logging to wandb**: runner resumes the wandb run from the subprocess and logs UMAP images + benchmark metrics. Chain: `FinetuneResult.wandb_run_id` -> npz -> `subprocess_embed` -> runner resumes with `wandb.init(id=..., resume="must")`.
