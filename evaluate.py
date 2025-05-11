import os
import csv
import re
import subprocess
from pathlib import Path

BASE_PATH  = Path("/mnt/d/LLM")
ENGINE_DIR = BASE_PATH / "trt_engines"
MODEL_DIR  = BASE_PATH / "models"
MMLU_PATH  = BASE_PATH / "mmlu"

RESULT_CSV = BASE_PATH / "benchmark_results.csv"
LOG_DIR    = BASE_PATH / "results"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODELS   = ["distilgpt2", "gpt2", "phi-2", "Llama-2-7b-hf"]
FORMATS  = ["int8_wo", "full_prec", "int4_awq"]
BATCH_SZ = 16

latency_re   = re.compile(r"latency\(ms\)\s*[:=]?\s*([\d.]+)", re.I)
tokens_re    = re.compile(r"(tokens_per_sec|tokens/sec|throughput)\s*[:=]?\s*([\d.]+)", re.I)
memory_re    = re.compile(r"gpu_peak_mem\(gb\)\s*[:=]?\s*([\d.]+)", re.I)

weighted_acc_re = re.compile(r"weighted average accuracy[:=]?\s*([\d.]+)", re.I)
subject_acc_re  = re.compile(r"average accuracy\s*([\d.]+)\s*-\s*\w+", re.I)

rouge1_re    = re.compile(r"rouge1:\s*([\d.]+)", re.I)
rouge2_re    = re.compile(r"rouge2:\s*([\d.]+)", re.I)
rougeL_re    = re.compile(r"rougeL:\s*([\d.]+)", re.I)
rougeLsum_re = re.compile(r"rougeLsum:\s*([\d.]+)", re.I)

def run(cmd: list[str], logfile: Path) -> tuple[int, str]:
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    logfile.write_text(res.stdout, encoding="utf-8")
    return res.returncode, res.stdout

def parse_benchmark(output: str) -> tuple[str, str, str]:
    lat = latency_re.search(output)
    tps = tokens_re.search(output)
    mem = memory_re.search(output)
    return (
        float(lat.group(1)) if lat else "",
        float(tps.group(2)) if tps else "",
        float(mem.group(1)) * 1024 if mem else ""
    )

def parse_mmlu(output: str) -> str:
    acc_match = weighted_acc_re.search(output) or subject_acc_re.search(output)
    acc = float(acc_match.group(1)) if acc_match else ""
    if acc != "" and acc <= 1.0:
        acc *= 100
    return acc

def parse_rouge(output: str) -> tuple[str, str, str, str]:
    r1  = float(rouge1_re.search(output).group(1)) if rouge1_re.search(output) else ""
    r2  = float(rouge2_re.search(output).group(1)) if rouge2_re.search(output) else ""
    rl  = float(rougeL_re.search(output).group(1)) if rougeL_re.search(output) else ""
    rls = float(rougeLsum_re.search(output).group(1)) if rougeLsum_re.search(output) else ""
    return r1, r2, rl, rls

csv_exists = RESULT_CSV.exists()
with RESULT_CSV.open("a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    if not csv_exists:
        writer.writerow([
            "model", "precision", "batch_size",
            "latency_ms", "tokens_per_sec", "gpu_memory_mb",
            "mmlu_accuracy", "rouge1", "rouge2", "rougeL", "rougeLsum"
        ])

    for model in MODELS:
        for fmt in FORMATS:
            tag = f"{model}_{fmt}"
            print(f"\nEvaluating: {tag}")

            engine_path = ENGINE_DIR / tag
            model_path  = MODEL_DIR / model

            bench_cmd = [
                "python3",
                str(BASE_PATH / "TensorRT-LLM/benchmarks/python/benchmark.py"),
                "--batch_size", str(BATCH_SZ),
                "--engine_dir", str(engine_path)
            ]
            bench_log = LOG_DIR / f"{tag}_benchmark.txt"
            rc, bench_out = run(bench_cmd, bench_log)
            if rc != 0:
                print(f"benchmark.py failed. See: {bench_log.name}")
                continue
            latency, tps, mem = parse_benchmark(bench_out)

            mmlu_cmd = [
                "python3",
                str(BASE_PATH / "TensorRT-LLM/examples/mmlu.py"),
                "--test_trt_llm",
                "--hf_model_dir", str(model_path),
                "--engine_dir",   str(engine_path),
                "--data_dir",     str(MMLU_PATH)
            ]
            mmlu_log = LOG_DIR / f"{tag}_mmlu.txt"
            rc, mmlu_out = run(mmlu_cmd, mmlu_log)
            if rc != 0:
                print(f"mmlu.py failed. See: {mmlu_log.name}")
                continue
            acc = parse_mmlu(mmlu_out)

            summarize_cmd = [
                "python3",
                str(BASE_PATH / "TensorRT-LLM/examples/summarize.py"),
                "--test_trt_llm",
                "--hf_model_dir", str(model_path),
                "--data_type", "fp16",
                "--engine_dir", str(engine_path)
            ]
            summarize_log = LOG_DIR / f"{tag}_summarize.txt"
            rc, summarize_out = run(summarize_cmd, summarize_log)
            if rc != 0:
                print(f"summarize.py failed. See: {summarize_log.name}")
                continue
            rouge1, rouge2, rougeL, rougeLsum = parse_rouge(summarize_out)

            writer.writerow([
                model, fmt, BATCH_SZ,
                latency, tps, mem,
                acc, rouge1, rouge2, rougeL, rougeLsum
            ])
            csvfile.flush()

            print(f"{tag} → Latency {latency} ms | Tokens per second {tps} | Accuracy {acc}% | ROUGE-L {rougeL}")
