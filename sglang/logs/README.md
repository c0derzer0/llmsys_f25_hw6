## SGLang Inference Experiments (Qwen2.5)

### Model Choice and RadixAttention
For the SGLang part, I used the **`Qwen/Qwen2.5-7B-Instruct`** model (non‑1M context version) instead of `Qwen/Qwen2.5-7B-Instruct-1M`.
This choice was intentional:
- The `...-1M` variant enables `dual_chunk_attention_config` in its `config.json`, which forces SGLang to use the `dual_chunk_flash_attn` backend and **disables RadixAttention**.
- By using the non‑1M instruct model, SGLang can use its standard FlashInfer backend with **RadixAttention** enabled, matching the assignment’s focus on cache reuse.

An alternative approach (mentioned on Ed) would be to:
- Download the `...-1M` model locally, and
- Manually edit its Hugging Face cache `config.json` to remove the `dual_chunk_attention_config` block so that dual‑chunk attention is not used.

I **did not** modify the cached config; instead, I chose the non‑long‑context model to keep the setup simple and explicit.

### Hardware and GPU Utilization Evidence
All SGLang runs were executed on a node with **2×NVIDIA A100 80 GB** GPUs.
To demonstrate that both GPUs were fully utilized during inference:
- I captured **`nvidia-smi` logs** showing that both A100s are active and that the SGLang inference process uses essentially the **entire available GPU memory** on each device.
- I stored the **logs** and **generated outputs** in this `sglang/logs` directory as evidence:
  - `training.logs` contains the SGLang run output and timing information.
  - I also saved `nvidia-smi` snapshots to show per‑GPU memory usage and utilization during inference.

Together, these artifacts show that:
- Inference is using **both GPUs**,
- The model plus KV cache utilize almost all GPU memory during the run,
- And the configuration is consistent with the assignment’s intent to demonstrate high‑utilization inference with RadixAttention.

