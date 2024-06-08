import os
import matplotlib.pyplot as plt
import numpy as np

# Directory containing the trace datasets
trace_directory = 'traces'
figure_directory = './traces/figures'
def plot_trace(Dataset_name,filename, timestamps, bandwidths):
    # Plot the trace data
    save_path = figure_directory + '/'+Dataset_name + '/' +'plots/'+filename
    if not os.path.exists(figure_directory + '/'+Dataset_name + '/' +'plots/'):
        os.makedirs(figure_directory + '/'+Dataset_name + '/' +'plots/')
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, bandwidths, label='Trace Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Bandwidth (Mbps)')
    plt.title(f'Trace Data: {filename}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_path}.png')
    plt.show()
    plt.close()

# Create the figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Dictionary to store the data for each dataset
dataset_data = {
    'fcc18': [],
    'ghent': [],
    'hsr': [],
    'lab': [],
    'oboe': []
}

# Iterate over each dataset directory
for dataset in dataset_data.keys():
    dataset_dir = os.path.join(trace_directory, dataset)
    
    # Check if the dataset directory exists
    if not os.path.exists(dataset_dir):
        print(f"Directory not found: {dataset_dir}")
        continue
    
    # Iterate over each trace file in the dataset directory
    trace_files = os.listdir(dataset_dir)
    
    if not trace_files:
        print(f"No trace files found in directory: {dataset_dir}")
        continue
    
    for filename in trace_files:
        timestamps = []
        bandwidths = []
        
        # Read the data from the file and skip directory 
        with open(os.path.join(dataset_dir, filename), 'r') as file:
            for line in file:
                try:
                    timestamp, bandwidth = line.strip().split()
                    timestamps.append(float(timestamp))
                    bandwidths.append(float(bandwidth))
                except ValueError:
                    print(f"Invalid data format in file: {filename}")
                    continue
        
        # plot the trace data
        plot_trace(dataset,filename,timestamps, bandwidths)
        # Append the data to the respective dataset list
        dataset_data[dataset].append((timestamps, bandwidths))

# Check if any data was loaded
if not any(dataset_data.values()):
    print("No data found in the trace files.")
    exit(1)

# Plot the CDF for average bandwidth and standard deviation for each dataset
Max_avg_bandwidth = 0
Max_std_dev = 0
for dataset, data in dataset_data.items():
    if not data:
        continue

    # Calculate average bandwidth and standard deviation for each trace in the dataset
    avg_bandwidths = []
    std_devs = []
    for _, bandwidths in data:
        avg_bandwidths.append(sum(bandwidths) / len(bandwidths))
        variance = sum((x - avg_bandwidths[-1]) ** 2 for x in bandwidths) / len(bandwidths)
        std_devs.append(variance ** 0.5)
    
    # Plot CDF for average bandwidth
    sorted_avg_bandwidths = np.sort(avg_bandwidths)
    if Max_avg_bandwidth < max(sorted_avg_bandwidths):
        Max_avg_bandwidth = max(sorted_avg_bandwidths)
    cdf_avg = np.arange(len(sorted_avg_bandwidths)) / float(len(sorted_avg_bandwidths))
    ax1.plot(sorted_avg_bandwidths, cdf_avg * 100, linestyle='-', marker='o',markersize=1.5, label=dataset)
    
    # Plot CDF for standard deviation
    sorted_std_devs = np.sort(std_devs)
    if Max_std_dev < max(sorted_std_devs):
        Max_std_dev = max(sorted_std_devs)
    cdf_std = np.arange(len(sorted_std_devs)) / float(len(sorted_std_devs))
    
    ax2.plot(sorted_std_devs, cdf_std * 100, linestyle='-', marker='o',markersize=1.5, label=dataset)

# Set labels and limits for the average bandwidth CDF plot
ax1.set_xlabel('Average Bandwidth (Mbps)')
ax1.set_ylabel('CDF (%)')
ax1.set_xlim(0, Max_avg_bandwidth)
ax1.set_ylim(0, 100)
ax1.grid(True)
ax1.legend()

# Set labels and limits for the standard deviation CDF plot
ax2.set_xlabel('Standard Deviation (Mbps)')
ax2.set_ylabel('CDF (%)')
ax2.set_xlim(0, Max_std_dev)
ax2.set_ylim(0, 100)
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig('cdf_plots.png')
plt.show()