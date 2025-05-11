import subprocess
import sys
from pathlib import Path

CKPT_DIR = Path("/mnt/d/LLM/trt_ckpt")
ENGINE_DIR = Path("/mnt/d/LLM/trt_engines")

BATCH_SIZE = 16
MAX_INPUT_LEN = 2048
MAX_SEQ_LEN = 4096

MODELS = ["distilgpt2", "gpt2", "phi-2", "Llama-2-7b-hf"]
FORMATS = ["full_prec", "int4_awq", "int8_wo"]

def model_params(model, fmt):
    params = []
    if model == "Llama-2-7b-hf":
        params.extend([
            "--max_input_len", str(MAX_INPUT_LEN),
            "--max_seq_len", str(MAX_SEQ_LEN)
        ])
    elif model == "phi-2":
        params.extend(["--max_input_len", "1024", "--max_seq_len", "2048"])
    return params

for model in MODELS:
    for fmt in FORMATS:
        tag = f"{model}_{fmt}"
        ckpt_path = CKPT_DIR / tag
        engine_path = ENGINE_DIR / tag
        engine_path.mkdir(parents=True, exist_ok=True)
        print(f"Building engine for {model} [{fmt}]")
        cmd = [
            "trtllm-build",
            "--checkpoint_dir", str(ckpt_path),
            "--output_dir", str(engine_path),
            "--gemm_plugin", "float16",
            "--max_batch_size", str(BATCH_SIZE),
        ]
        cmd.extend(model_params(model, fmt))
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            sys.stdout.write(line)
        process.wait()
        if process.returncode == 0:
            print(f"Engine built successfully for {tag}")
        else:
            print(f"Build failed for {tag}")

