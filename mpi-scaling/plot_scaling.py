import matplotlib.pyplot as plt

processes = [1, 2, 4]
measured_time = [35.6466, 17.9084, 9.0653]
measured_speedup = [1.0, 1.99, 3.93]
ideal_speedup = [1, 2, 4]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Chart 1: Raw time
axes[0].plot(processes, measured_time, marker='o', color='#d9534f', linewidth=2)
axes[0].set_xlabel('Number of MPI Processes')
axes[0].set_ylabel('Time (seconds)')
axes[0].set_title('Wall-Clock Time vs. Process Count')
axes[0].set_xticks(processes)
for x, y in zip(processes, measured_time):
    axes[0].annotate(f'{y:.1f}s', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

# Chart 2: Speedup vs ideal
axes[1].plot(processes, ideal_speedup, marker='o', linestyle='--', color='gray', label='Ideal (linear)')
axes[1].plot(processes, measured_speedup, marker='o', color='#5cb85c', linewidth=2, label='Measured')
axes[1].set_xlabel('Number of MPI Processes')
axes[1].set_ylabel('Speedup vs. 1 process')
axes[1].set_title('Speedup: Measured vs. Ideal')
axes[1].set_xticks(processes)
axes[1].legend()
for x, y in zip(processes, measured_speedup):
    axes[1].annotate(f'{y:.2f}x', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

plt.suptitle('MPI Scaling: Monte Carlo Pi Estimation (Oracle Cloud A1, 4 real cores)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/scaling_results.png', dpi=150)
print("Saved plot to plots/scaling_results.png")
