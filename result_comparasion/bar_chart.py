import pandas as pd
import matplotlib.pyplot as plt

# Load all the datasets
video_game_performance = pd.read_csv('video_game_performance_comparison.csv')
bbb_performance = pd.read_csv('BBB_360p24_performance_comparison.csv')
lol_performance = pd.read_csv('LOL_3D_performance_comparison.csv')
sport_highlight_performance = pd.read_csv('sport_highlight_performance_comparison.csv')
sport_long_take_performance = pd.read_csv('sport_long_take_performance_comparison.csv')
tos_performance = pd.read_csv('TOS_360p24_performance_comparison.csv')
underwater_performance = pd.read_csv('underwater_performance_comparison.csv')

datasets = [
    video_game_performance,
    bbb_performance,
    lol_performance,
    sport_highlight_performance,
    sport_long_take_performance,
    tos_performance,
    underwater_performance
]

# Define colors for each method
colors = {
    'Our Method': '#A7C7E7',
    'Baseline': '#E8C3A4',
    'GOP-5': '#B0C4B1',
    'Constant-5': '#D9A7A7'
}

# Function to filter data to 5-95 percentile
def filter_percentile(df, lower=5, upper=95):
    return df[(df >= df.quantile(lower / 100)) & (df <= df.quantile(upper / 100))]

# Filter each dataset to 5-95 percentile
filtered_datasets = []
for dataset in datasets:
    filtered = dataset.copy()
    for column in dataset.select_dtypes(include='number').columns:
        filtered[column] = filter_percentile(dataset[column])
    filtered_datasets.append(filtered.dropna())

# Calculate mean performance metrics for each method across all filtered datasets
filtered_mean_performance = pd.DataFrame()

for dataset in filtered_datasets:
    dataset_mean = dataset.groupby('Method').mean(numeric_only=True).reset_index()
    filtered_mean_performance = pd.concat([filtered_mean_performance, dataset_mean])
filtered_mean_performance = filtered_mean_performance.groupby('Method').mean(numeric_only=True).reset_index()

# Plot the filtered mean performance for each method
fig, axes = plt.subplots(3, 2, figsize=(15, 18))

axes[0, 0].bar(filtered_mean_performance['Method'], filtered_mean_performance['Average VMAF Score'], 
               color=[colors[method] for method in filtered_mean_performance['Method']])
axes[0, 0].set_title('Average VMAF Score')
axes[0, 0].set_xlabel('Method')
axes[0, 0].set_ylabel('VMAF Score')

axes[0, 1].bar(filtered_mean_performance['Method'], filtered_mean_performance['Stall Ratio(%)'], 
               color=[colors[method] for method in filtered_mean_performance['Method']])
axes[0, 1].set_title('Stall Ratio(%)')
axes[0, 1].set_xlabel('Method')
axes[0, 1].set_ylabel('Stall Ratio(%)')

axes[1, 0].bar(filtered_mean_performance['Method'], filtered_mean_performance['Average Size(MB)'], 
               color=[colors[method] for method in filtered_mean_performance['Method']])
axes[1, 0].set_title('Average Size(MB)')
axes[1, 0].set_xlabel('Method')
axes[1, 0].set_ylabel('Size(MB)')

axes[1, 1].bar(filtered_mean_performance['Method'], filtered_mean_performance['Total Size(MB)'], 
               color=[colors[method] for method in filtered_mean_performance['Method']])
axes[1, 1].set_title('Total Size(MB)')
axes[1, 1].set_xlabel('Method')
axes[1, 1].set_ylabel('Size(MB)')

axes[2, 0].bar(filtered_mean_performance['Method'], filtered_mean_performance['Average Bitrate(bps)'], 
               color=[colors[method] for method in filtered_mean_performance['Method']])
axes[2, 0].set_title('Average Bitrate(bps)')
axes[2, 0].set_xlabel('Method')
axes[2, 0].set_ylabel('Bitrate(bps)')

axes[2, 1].bar(filtered_mean_performance['Method'], filtered_mean_performance['Switch Ratio(%)'], 
               color=[colors[method] for method in filtered_mean_performance['Method']])
axes[2, 1].set_title('Switch Ratio(%)')
axes[2, 1].set_xlabel('Method')
axes[2, 1].set_ylabel('Switch Ratio(%)')

plt.tight_layout()
plt.show()
plt.savefig('performance_metrics.png')
