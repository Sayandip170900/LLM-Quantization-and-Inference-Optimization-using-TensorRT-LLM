# LLM Quantization and Inference Optimization using TensorRT-LLM

This repository benchmarks multiple Large Language Models in different precision formats using TensorRT-LLM. The workflow includes quantization, engine building, inference benchmarking, accuracy evaluation, and result visualization.

> Note: All experiments were conducted on an **NVIDIA RTX A4000 (Ampere architecture)**. This GPU does **not support FP8** execution, so FP8 benchmarking was excluded from this project.

---

## Project Structure

### Scripts

- `quantize.py`: Converts HuggingFace models into quantized checkpoint format (`int8_wo`, `int4_awq`, `full_prec`).
- `engine.py`: Builds TensorRT-LLM engines from those checkpoints for supported models and precisions.
- `evaluate.py`: Benchmarks each engine (latency, tokens/sec, memory), evaluates MMLU accuracy and ROUGE scores.
- `plot.py`: Generates a single 2×4 grid comparison chart of 8 metrics across all models and precisions.

---

## Models and Formats

### Evaluated Models
- `distilgpt2`
- `gpt2`
- `phi-2`
- `Llama-2-7b`

### Precision Formats
- `int8_wo`: Int8 weight-only quantization
- `int4_awq`: Int4 activation-aware quantization
- `full_prec`: Full precision (fp16)

---

## Metrics Evaluated

- **Latency (ms)**: Average generation time per request
- **Tokens per second**: Throughput efficiency
- **GPU memory (MB)**: Peak memory usage
- **MMLU Accuracy (%)**: Accuracy over a standardized QA benchmark
- **ROUGE-1 / ROUGE-2 / ROUGE-L / ROUGE-Lsum**: Generation quality compared to reference summaries

---

## Benchmark Table

| Model         | Precision | Batch Size | Latency (ms) | Tokens/sec | GPU Memory (MB) | Accuracy (%) | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|---------------|-----------|------------|--------------|-------------|------------------|---------------|----------|----------|----------|-------------|
| distilgpt2    | int8_wo   | 16         | 43.03        | 7437.19     | 1347.58          | 24.79         | 19.32    | 6.19     | 14.26    | 17.42       |
| distilgpt2    | full_prec | 16         | 47.78        | 6696.80     | 1463.30          | 24.87         | 17.41    | 5.42     | 13.29    | 15.55       |
| distilgpt2    | int4_awq  | 16         | 44.00        | 7272.07     | 1402.88          | 4.39          | 0.00     | 0.00     | 0.00     | 0.00        |
| gpt2          | int8_wo   | 16         | 60.83        | 5260.82     | 1540.10          | 26.29         | 17.40    | 4.44     | 12.90    | 15.33       |
| gpt2          | full_prec | 16         | 64.19        | 4985.36     | 1542.14          | 26.15         | 15.60    | 3.93     | 11.45    | 13.41       |
| gpt2          | int4_awq  | 16         | 59.63        | 5366.61     | 1607.68          | 67.00         | 0.62     | 0.00     | 0.66     | 0.62        |
| phi-2         | int8_wo   | 16         | 437.55       | 731.35      | 5262.34          | 56.68         | 26.69    | 8.37     | 17.28    | 23.72       |
| phi-2         | full_prec | 16         | 581.79       | 550.03      | 7809.02          | 56.77         | 29.41    | 10.22    | 18.87    | 25.61       |
| phi-2         | int4_awq  | 16         | 407.26       | 785.73      | 4136.96          | 54.08         | 23.90    | 8.07     | 17.52    | 20.80       |
| Llama-2-7b-hf | int8_wo   | 16         | 986.58       | 324.35      | 9701.06          | 45.96         | 26.79    | 10.55    | 19.28    | 23.46       |
| Llama-2-7b-hf | full_prec | 16         | 1456.40      | 219.72      | 15870.98         | 45.92         | 27.61    | 10.47    | 19.15    | 24.00       |
| Llama-2-7b-hf | int4_awq  | 16         | 900.51       | 355.35      | 6711.30          | 43.40         | 27.32    | 9.36     | 18.25    | 24.79       |

## Visual Summary

The following figure shows side-by-side comparisons of all metrics across models and quantization formats:

![Benchmark Plot](./plot.png)

Each bar chart represents one metric, comparing the three precisions per model.

---

## Key Observations

- **Latency and tokens/sec** improve significantly with quantization, especially in `int8_wo`.
- **ROUGE scores** remain relatively stable for larger models (phi-2, Llama-2-7b), but degrade for small models under `int4_awq`.
- **distilgpt2** and **gpt2** produced zero or poor ROUGE output under int4 quantization, indicating quantization loss or decoding failures.
- **phi-2** and **Llama-2-7b** are more robust to quantization, especially `int8_wo`.

---

## Conclusion

The results demonstrate that **weight-only INT8 quantization (`int8_wo`) is the most practical and stable format** for accelerating LLM inference on the RTX A4000. It offers **substantial performance gains** (up to 30% faster tokens/sec) while **preserving both accuracy and generation quality**, particularly for medium and large models.

While **INT4-AWQ** can reduce latency and memory further, it is **less reliable** for smaller models such as `distilgpt2` and `gpt2`, which failed to generate valid outputs. However, larger architectures like `phi-2` and `Llama-2-7b-hf` handled INT4 better, albeit with some quality loss.

For deployment on non-FP8 GPUs like Ampere, **full precision (fp16)** remains a strong baseline, but quantization—especially INT8 weight-only—is highly recommended for balancing throughput and model fidelity.

---

## Output Files

- `benchmark_results.csv`: Full metrics table
- `plot.png`: Consolidated visualization