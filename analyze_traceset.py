import matplotlib.pyplot as plt
import numpy as np

# Average data points for each algorithm
stall_ratio = {
    'Rate-based': 3.5,
    'BBA': 3.4,
    'BOLA': 3.3,
    'Jade': 3.2,
    'RobustMPC': 3.1,
    'Fugu': 3.0,
    'Pensieve': 2.9,
    'Comyco': 2.8
}

vmaf_scores = {
    'Rate-based': 55,
    'BBA': 56,
    'BOLA': 57,
    'Jade': 58,
    'RobustMPC': 59,
    'Fugu': 60,
    'Pensieve': 61,
    'Comyco': 62
}

# Errors (confidence intervals)
stall_ratio_errors = {
    'Rate-based': 0.04,
    'BBA': 0.1,
    'BOLA': 0.02,
    'Jade': 0.1,
    'RobustMPC': 0.1,
    'Fugu': 0.03,
    'Pensieve': 0.05,
    'Comyco': 0.1
}

vmaf_errors = {
    'Rate-based': 3,
    'BBA': 6,
    'BOLA': 2,
    'Jade': 1,
    'RobustMPC': 4,
    'Fugu': 2,
    'Pensieve': 3,
    'Comyco': 3
}


quality_smoothness = [16, 14, 12, 10, 8]
vmaf_vs_vmaf_change = {
    'Rate-based': [45, 50, 55, 60, 65],
    'BBA': [46, 51, 56, 61, 66],
    'BOLA': [47, 52, 57, 62, 67],
    'Jade': [48, 53, 58, 63, 68],
    'RobustMPC': [49, 54, 59, 64, 69],
    'Fugu': [50, 55, 60, 65, 70],
    'Pensieve': [51, 56, 61, 66, 71],
    'Comyco': [52, 57, 62, 67, 72]
}
vmaf_vs_vmaf_change_err = {algo: [1, 1, 1, 1, 1] for algo in vmaf_vs_vmaf_change}

buffer_sizes = [25, 20, 15, 10]
qoe_dnn_vs_buffer = {
    'Rate-based': [-2.5, -2.0, -1.5, -1.0],
    'BBA': [-2.4, -1.9, -1.4, -0.9],
    'BOLA': [-2.3, -1.8, -1.3, -0.8],
    'Jade': [-2.2, -1.7, -1.2, -0.7],
    'RobustMPC': [-2.1, -1.6, -1.1, -0.6],
    'Fugu': [-2.0, -1.5, -1.0, -0.5],
    'Pensieve': [-1.9, -1.4, -0.9, -0.4],
    'Comyco': [-1.8, -1.3, -0.8, -0.3]
}
qoe_dnn_vs_buffer_err = {algo: [0.1, 0.1, 0.1, 0.1] for algo in qoe_dnn_vs_buffer}

qoe_dnn_values = np.linspace(-3, 1, 100)
cdf_qoe_dnn = {
    'BBA': np.random.rand(100),
    'BOLA': np.random.rand(100),
    'RobustMPC': np.random.rand(100),
    'Pensieve': np.random.rand(100),
    'Comyco': np.random.rand(100),
    'Jade': np.random.rand(100)
}

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot (a) VMAF vs. Stall Ratio
markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p']
for name, marker in zip(stall_ratio.keys(), markers):
    x = stall_ratio[name]
    y = vmaf_scores[name]
    x_err = stall_ratio_errors[name]
    y_err = vmaf_errors[name]
    axs[0,0].errorbar(x, y, xerr=x_err, yerr=y_err, label=name, marker=marker, capsize=5)


axs[0, 0].set_xlabel('Time Spent on Stall (%)')
axs[0, 0].set_ylabel('Video Quality (VMAF)')
axs[0, 0].set_title('(a) VMAF vs. Stall Ratio')
axs[0, 0].legend()
axs[0, 0].grid(True)
axs[0, 0].invert_xaxis()

# Plot (b) VMAF vs. VMAF Change
for algo, vmaf_values in vmaf_vs_vmaf_change.items():
    axs[0, 1].errorbar(quality_smoothness, vmaf_values, yerr=vmaf_vs_vmaf_change_err[algo], marker='o', capsize=4, label=algo)
axs[0, 1].set_xlabel('Quality Smoothness (VMAF)')
axs[0, 1].set_ylabel('Video Quality (VMAF)')
axs[0, 1].set_title('(b) VMAF vs. VMAF Change')
axs[0, 1].legend()

# Plot (c) QoE_DNN vs. Buffer
for algo, qoe_values in qoe_dnn_vs_buffer.items():
    axs[1, 0].errorbar(buffer_sizes, qoe_values, yerr=qoe_dnn_vs_buffer_err[algo], marker='o', capsize=4, label=algo)
axs[1, 0].set_xlabel('Buffer (s)')
axs[1, 0].set_ylabel('QoE')
axs[1, 0].set_title('(c) QoE_DNN vs. Buffer')
axs[1, 0].legend()

# Plot (d) CDF of QoE_DNN


# Generating sample data: Normal distribution of QoE around different means
np.random.seed(0)
data = {
    'BBA': np.random.normal(-1, 1, 1000),
    'BOLA': np.random.normal(-0.5, 1, 1000),
    'RobustMPC': np.random.normal(0, 1, 1000),
    'Pensieve': np.random.normal(0.5, 1, 1000),
    'Comyco': np.random.normal(0, 1, 1000),
    'Jade': np.random.normal(-0.2, 1, 1000)
}

# Function to calculate the CDF
def plot_cdf(data, ax, label, linestyle):
    sorted_data = np.sort(data)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
    ax.plot(sorted_data, yvals, label=label, linestyle=linestyle)

# Plotting
linestyles = ['--', '-', ':', '-.', '--', '-']
colors = ['green', 'blue', 'black', 'magenta', 'cyan', 'red']

for (alg, vals), ls, color in zip(data.items(), linestyles, colors):
    plot_cdf(vals, axs[1,1], alg, ls)
axs[1, 1].set_xlabel('QoE')
axs[1, 1].set_ylabel('CDF')
axs[1, 1].set_title('(d) CDF of QoE_DNN')
axs[1, 1].legend()

# Adjust spacing between subplots
plt.tight_layout()
# Save the plot as an image
plt.savefig('analyze_traceset.png')

# Display the plot
plt.show()