import matplotlib.pyplot as plt
import numpy as np

# Sample data for each subplot (replace with your own data)
stall_ratio = [4.0, 3.5, 3.0, 2.5, 2.0]
vmaf_vs_stall = {
    'Rate-based': [45, 50, 55, 60, 65],
    'BBA': [46, 51, 56, 61, 66],
    'BOLA': [47, 52, 57, 62, 67],
    'Jade': [48, 53, 58, 63, 68],
    'RobustMPC': [49, 54, 59, 64, 69],
    'Fugu': [50, 55, 60, 65, 70],
    'Pensieve': [51, 56, 61, 66, 71],
    'Comyco': [52, 57, 62, 67, 72]
}
vmaf_vs_stall_err_score = {algo: [5,5,5,5,5] for algo in vmaf_vs_stall}
vmaf_vs_stall_err_time = {algo: [0.1, 0.1, 0.2, 0.1, 0.1] for algo in vmaf_vs_stall}

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
for algo, vmaf_values in vmaf_vs_stall.items():
    # set marker for each algorithm to distinguish them
    marker = markers.pop(0)
    # plot the data points with error bars for each algorithm ,xerr is the error in x-axis 
    # axs[0,0].plot(stall_ratio, vmaf_values, marker=marker, label=algo)
    axs[0, 0].errorbar(stall_ratio, vmaf_values, \
                       yerr=vmaf_vs_stall_err_score[algo],\
                       xerr=vmaf_vs_stall_err_time[algo],\
                       linestyle='',\
                       #fmt='', \
                       capsize=4,label = algo, marker=marker)
axs[0, 0].set_xlabel('Time Spent on Stall (%)')
axs[0, 0].set_ylabel('Video Quality (VMAF)')
axs[0, 0].set_title('(a) VMAF vs. Stall Ratio')
axs[0, 0].legend()
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
for algo, cdf_values in cdf_qoe_dnn.items():
    axs[1, 1].plot(qoe_dnn_values, cdf_values, label=algo)
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