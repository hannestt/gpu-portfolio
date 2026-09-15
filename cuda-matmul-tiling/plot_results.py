import matplotlib.pyplot as plt

# Our measured results from Nsight Compute
labels = ['Naive', 'Tiled (shared memory)']
duration_ms = [9.04, 5.78]
compute_throughput_pct = [63.55, 74.51]
pct_of_peak_fp32 = [8, 12]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Chart 1: Duration (lower is better)
axes[0].bar(labels, duration_ms, color=['#d9534f', '#5cb85c'])
axes[0].set_ylabel('Duration (ms)')
axes[0].set_title('Kernel Duration (1024x1024 matmul)')
for i, v in enumerate(duration_ms):
    axes[0].text(i, v + 0.15, f'{v}ms', ha='center', fontweight='bold')

# Chart 2: Compute throughput (higher is better)
axes[1].bar(labels, compute_throughput_pct, color=['#d9534f', '#5cb85c'])
axes[1].set_ylabel('Compute (SM) Throughput (%)')
axes[1].set_title('SM Throughput')
axes[1].set_ylim(0, 100)
for i, v in enumerate(compute_throughput_pct):
    axes[1].text(i, v + 2, f'{v}%', ha='center', fontweight='bold')

# Chart 3: % of peak fp32 performance (higher is better)
axes[2].bar(labels, pct_of_peak_fp32, color=['#d9534f', '#5cb85c'])
axes[2].set_ylabel('% of Peak FP32 Performance')
axes[2].set_title('Roofline: % of Peak Achieved')
axes[2].set_ylim(0, 20)
for i, v in enumerate(pct_of_peak_fp32):
    axes[2].text(i, v + 0.3, f'{v}%', ha='center', fontweight='bold')

plt.suptitle('Naive vs Tiled Matrix Multiply — 1.56x Speedup (Tesla T4)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('/content/gpu-portfolio/cuda-matmul-tiling/plots/naive_vs_tiled.png', dpi=150)
print("Saved plot to plots/naive_vs_tiled.png")
