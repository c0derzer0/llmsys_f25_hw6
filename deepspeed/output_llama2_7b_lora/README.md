## DeepSpeed Configuration Exploration

### Goal
Emulate training LLaMA-2-7B with LoRA on a 2×V100 16 GB setup, using the available 2×A100 80 GB GPUs, while:
- Keeping per-GPU memory usage in under 16 GB.
- Maintaining stable training and improving evaluation perplexity across epochs

### Hyperparameter Sweeps
I explored several combinations of **max sequence length**, **LoRA rank**, and **per-device batch size**:
- **Max sequence length**: 384, 412, 448, 512
- **LoRA dimension**: 64 and 128
- **Per-device train batch size**: 1, 2, 4

For each configuration, I:
- Monitored **GPU memory usage and utilization** using `nvidia-smi` (logged to CSV and summarized with a small analysis script)
- Tracked **training/evaluation loss and perplexity** with Weights & Biases (W&B), logging:
  - `train/loss` periodically during the training loop
  - `eval/loss` and `eval/perplexity` at the initial evaluation and after each epoch

### Final Chosen Configuration
Based on these experiments, I selected the following configuration as a good trade-off between utilization and stability:
- **LoRA dimension**: 64
- **Per-device train batch size**: 4
- **Per-device eval batch size**: 4
- **Max sequence length**: 432
- **Gradient accumulation steps**: 8
- **ZeRO stage**: 3
- **dtype**: `bf16`

With this setup, `nvidia-smi` reports per-GPU memory usage of roughly 15.5 GB during steady-state training, and both GPUs reach close to 100 % compute utilization.

### Evaluation Metrics (Perplexity and Loss)
Using the final configuration, I observed the following evaluation results (from the DeepSpeed logs):
- **Epoch 0/2**: perplexity ≈ 6.26, loss ≈ 1.83
- **Epoch 1/2**: perplexity ≈ 4.49, loss ≈ 1.50
- **Epoch 2/2**: perplexity ≈ 4.41, loss ≈ 1.48

These values show a consistent improvement in both loss and perplexity across epochs.

### Memory Usage Evidence
