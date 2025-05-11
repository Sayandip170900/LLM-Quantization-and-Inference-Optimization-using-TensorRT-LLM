import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/mnt/d/LLM/benchmark_results.csv")

metrics = [
    "latency_ms", "tokens_per_sec", "gpu_memory_mb", "accuracy_percent",
    "rouge1", "rouge2", "rougeL", "rougeLsum"
]

titles = [
    "Latency (ms)", "Tokens/sec", "GPU Memory (MB)", "MMLU Accuracy (%)",
    "ROUGE-1", "ROUGE-2", "ROUGE-L", "ROUGE-Lsum"
]

fig, axes = plt.subplots(2, 4, figsize=(32, 12))

for i, metric in enumerate(metrics):
    ax = axes[i // 4, i % 4]
    sns.barplot(data=df, x="model", y=metric, hue="precision", ax=ax)
    ax.set_title(titles[i], fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title="Precision", fontsize=10, title_fontsize=11)

plt.tight_layout()
plt.savefig("/mnt/d/LLM/plot.png")
plt.close()
