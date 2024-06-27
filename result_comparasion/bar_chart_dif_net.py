import pandas as pd
import matplotlib.pyplot as plt

# Reload the datasets
video_game_performance = pd.read_csv('video_game_performance_comparison.csv')
bbb_performance = pd.read_csv('BBB_360p24_performance_comparison.csv')
lol_performance = pd.read_csv('LOL_3D_performance_comparison.csv')
sport_highlight_performance = pd.read_csv('sport_highlight_performance_comparison.csv')
sport_long_take_performance = pd.read_csv('sport_long_take_performance_comparison.csv')
tos_performance = pd.read_csv('TOS_360p24_performance_comparison.csv')
underwater_performance = pd.read_csv('underwater_performance_comparison.csv')

datasets = [
    underwater_performance,
    video_game_performance,
    bbb_performance,
    lol_performance,
    sport_highlight_performance,
    sport_long_take_performance,
    tos_performance
]

# Define Morandi colors for each method
morandi_colors = {
    'Our Method': '#A7C7E7',
    'Baseline': '#E8C3A4',
    'GOP-5': '#B0C4B1',
    'Constant-5': '#D9A7A7'
}

# Classify traces based on CDF plots into Slow, Medium, Fast
# Manually classify traces based on provided CDF plot image information
slow_traces = [
'obeoe_trace', 'fcc18_trace'
]
medium_traces = [
'lab_trace', 'hsr_trace'
]
fast_traces = [
'ghent_trace'
]

# Function to classify each row based on the trace file name
def classify_trace(file_name):
    if any(trace in file_name for trace in slow_traces):
        return 'Slow'
    elif any(trace in file_name for trace in medium_traces):
        return 'Medium'
    elif any(trace in file_name for trace in fast_traces):
        return 'Fast'
    else:
        return 'Unknown'

# Add a column to classify each trace
for dataset in datasets:
    dataset['Network Condition'] = dataset['File'].apply(classify_trace)

# Filter the datasets based on network conditions
def filter_by_network_condition(datasets, condition):
    return [dataset[dataset['Network Condition'] == condition] for dataset in datasets]

slow_datasets = filter_by_network_condition(datasets, 'Slow')
medium_datasets = filter_by_network_condition(datasets, 'Medium')
fast_datasets = filter_by_network_condition(datasets, 'Fast')

# Function to calculate mean performance metrics for each method
def calculate_mean_performance(datasets):
    mean_performance = pd.DataFrame()
    for dataset in datasets:
        dataset_mean = dataset.groupby('Method').mean(numeric_only=True).reset_index()
        mean_performance = pd.concat([mean_performance, dataset_mean])
    return mean_performance.groupby('Method').mean(numeric_only=True).reset_index()

# Calculate mean performance for each network condition
slow_performance = calculate_mean_performance(slow_datasets)
medium_performance = calculate_mean_performance(medium_datasets)
fast_performance = calculate_mean_performance(fast_datasets)

# Plot performance for each network condition
fig, axes = plt.subplots(3, 2, figsize=(15, 18))

axes[0, 0].bar(slow_performance['Method'], slow_performance['Average VMAF Score'], 
               color=[morandi_colors[method] for method in slow_performance['Method']])
axes[0, 0].set_title('Slow Network: Average VMAF Score')
axes[0, 0].set_xlabel('Method')
axes[0, 0].set_ylabel('VMAF Score')

axes[0, 1].bar(slow_performance['Method'], slow_performance['Stall Ratio(%)'], 
               color=[morandi_colors[method] for method in slow_performance['Method']])
axes[0, 1].set_title('Slow Network: Stall Ratio(%)')
axes[0, 1].set_xlabel('Method')
axes[0, 1].set_ylabel('Stall Ratio(%)')

axes[1, 0].bar(medium_performance['Method'], medium_performance['Average VMAF Score'], 
               color=[morandi_colors[method] for method in medium_performance['Method']])
axes[1, 0].set_title('Medium Network: Average VMAF Score')
axes[1, 0].set_xlabel('Method')
axes[1, 0].set_ylabel('VMAF Score')

axes[1, 1].bar(medium_performance['Method'], medium_performance['Stall Ratio(%)'], 
               color=[morandi_colors[method] for method in medium_performance['Method']])
axes[1, 1].set_title('Medium Network: Stall Ratio(%)')
axes[1, 1].set_xlabel('Method')
axes[1, 1].set_ylabel('Stall Ratio(%)')

axes[2, 0].bar(fast_performance['Method'], fast_performance['Average VMAF Score'], 
               color=[morandi_colors[method] for method in fast_performance['Method']])
axes[2, 0].set_title('Fast Network: Average VMAF Score')
axes[2, 0].set_xlabel('Method')
axes[2, 0].set_ylabel('VMAF Score')

axes[2, 1].bar(fast_performance['Method'], fast_performance['Stall Ratio(%)'], 
               color=[morandi_colors[method] for method in fast_performance['Method']])
axes[2, 1].set_title('Fast Network: Stall Ratio(%)')
axes[2, 1].set_xlabel('Method')
axes[2, 1].set_ylabel('Stall Ratio(%)')

plt.tight_layout()
plt.show()
plt.savefig('bar_chart_dif_net.png')
