import matplotlib.pyplot as plt

operations = ['All-Reduce', 'All-Gather', 'Broadcast']
bandwidth = [1.89, 1.46, 2.04]
colors = ['#5cb85c', '#d9534f', '#5bc0de']

plt.figure(figsize=(8, 5))
bars = plt.bar(operations, bandwidth, color=colors)
plt.ylabel('Average Bus Bandwidth (GB/s)')
plt.title('NCCL Collective Operations Comparison\n(2x RTX 4090, PCIe-only, no NVLink)')
for bar, val in zip(bars, bandwidth):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val} GB/s', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('/root/gpu-portfolio/nccl-benchmark/plots/collective_comparison.png', dpi=150)
print("Saved chart")
