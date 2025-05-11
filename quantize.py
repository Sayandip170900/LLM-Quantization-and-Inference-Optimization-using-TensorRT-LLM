import os
import subprocess

QUANT_PATH = "/mnt/d/LLM/TensorRT-LLM/examples/quantization/quantize.py"
CALIB_BATCH_SIZE = 32
OUTPUT_DIR = "/mnt/d/LLM/trt_ckpt"
MODEL_ROOT = "/mnt/d/LLM/models"

MODELS = {
    "distilgpt2": os.path.join(MODEL_ROOT, "distilgpt2"),
    "gpt2": os.path.join(MODEL_ROOT, "gpt2"),
    "phi-2": os.path.join(MODEL_ROOT, "phi-2"),
    "Llama-2-7b-hf": os.path.join(MODEL_ROOT, "Llama-2-7b-hf"),
}

QFORMATS = ["int8_wo", "full_prec", "int4_awq"]

for model_name, model_path in MODELS.items():
    for qformat in QFORMATS:
        out_dir = os.path.join(OUTPUT_DIR, f"{model_name}_{qformat}")

        cmd = [
            "python3", QUANT_PATH,
            "--model_dir", model_path,
            "--batch_size", str(CALIB_BATCH_SIZE),
            "--qformat", qformat,
            "--output_dir", out_dir
        ]

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode == 0:
            print(f"Success: {model_name} [{qformat}]")
        else:
            print(f"Failed: {model_name} [{qformat}]")
            print(result.stderr)
