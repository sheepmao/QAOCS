import pandas as pd
import matplotlib.pyplot as plt


def plot_segment_duration(baseline_df, your_method_df):
    # Plot the segment duration for the baseline and your method in bar chart
    # X-axis: segment number, Y-axis: segment duration
    plt.figure(figsize=(10, 6))
    plt.bar(your_method_df['SEGMENT_NO'], your_method_df['DURATION']/1000, label='Your Method')
    plt.bar(baseline_df['SEGMENT_PROGRESSIVE'], baseline_df['DURATION']/1000, label='Baseline')
    plt.xlabel('Segment Number')
    plt.ylabel('Duration (seconds)')
    plt.title('Segment Duration Comparison')
    plt.legend()
    plt.savefig('segment_duration_comparison.png')
    plt.show()
def plot_segment_size(baseline_df, your_method_df):
    # Plot the segment size for the baseline and your method in bar chart
    # X-axis: time in secods, Y-axis: segment size in MB, bar width is duration
    x1=[]
    x2=[]
    for i in range(len(baseline_df)):
        x1.append(sum(baseline_df.loc[0:i, 'DURATION'])/1000)
    for i in range(len(your_method_df)):
        x2.append(sum(your_method_df.loc[0:i, 'DURATION'])/1000)
    plt.figure(figsize=(10, 6))
    plt.bar(x2, your_method_df['BYTES'] / 1e6, width=your_method_df['DURATION']/1000, label='Your Method',alpha=0.7)
    plt.bar(x1, baseline_df['BYTES'] / 1e6, width=baseline_df['DURATION']/1000, label='Baseline',alpha=0.35)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Size (MB)')
    plt.title('Segment Size Comparison')
    plt.legend()
    plt.savefig('segment_size_comparison.png')
    plt.show()

def plot_buffer_state(baseline_df, your_method_df):
    x1=[]
    x2=[]
    for i in range(len(baseline_df)):
        x1.append(sum(baseline_df.loc[0:i, 'DURATION'])/1000)
    for i in range(len(your_method_df)):
        x2.append(sum(your_method_df.loc[0:i, 'DURATION'])/1000)
    plt.figure(figsize=(10, 6))
    plt.plot(x2, your_method_df['BUFFER_STATE']/1000, label='Your Method')
    plt.plot(x1, baseline_df['BUFFER_STATE']/1000, label='Baseline')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Buffer State (seconds)')
    plt.title('Buffer State Comparison')
    plt.legend()
    plt.savefig('buffer_state_comparison.png')
    plt.show()
def plot_vmaf_score(baseline_df, your_method_df):
    # x-axis: segment duration, y-axis: VMAF score
    x1=[]
    x2=[]
    for i in range(len(baseline_df)):
        x1.append(sum(baseline_df.loc[0:i, 'DURATION'])/1000)
    for i in range(len(your_method_df)):
        x2.append(sum(your_method_df.loc[0:i, 'DURATION'])/1000)
    plt.figure(figsize=(10, 6))
    plt.plot(x2, your_method_df['VMAF'], label='Your Method')
    plt.plot(x1, baseline_df['VMAF'], label='Baseline')
    plt.xlabel('Time (seconds)')
    plt.ylabel('VMAF Score')
    plt.title('Quality (VMAF) Comparison')
    plt.legend()
    plt.savefig('vmaf_score_comparison.png')
    plt.show()
def plot_bitrate(baseline_df, your_method_df):
    x1=[]
    x2=[]
    for i in range(len(baseline_df)):
        x1.append(sum(baseline_df.loc[0:i, 'DURATION'])/1000)
    for i in range(len(your_method_df)):
        x2.append(sum(your_method_df.loc[0:i, 'DURATION'])/1000)
    plt.figure(figsize=(10, 6))
    plt.plot(x2, your_method_df['BITRATE'], label='Your Method')
    plt.plot(x1, baseline_df['BITRATE'], label='Baseline')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Bitrate (bps)')
    plt.title('Bitrate Comparison')
    plt.legend()
    plt.savefig('bitrate_comparison.png')
    plt.show()

def plot_size_boxplot(baseline_df, your_method_df):
    # Plot the size of the segments for the baseline and your method in a box plot in 5-95 percentile
    # X-axis: method, Y-axis: segment size in MB
    plt.figure(figsize=(10, 6))
    
    # Define colors for each box
    colors = ['#EB5E55', '#8F4899']  # Red and purple colors
    hatch_patterns = ['', '////']  # No hatch for 'Constant', hatch pattern for 'Time'
    
    bp = plt.boxplot([baseline_df['BYTES'] / 1e6, your_method_df['BYTES'] / 1e6],
                     labels=['Baseline', 'Your Method'],
                     whis=[5, 95],
                     patch_artist=True,
                     boxprops=dict(facecolor='white', edgecolor='black'),
                     widths=0.6)
    
    # Set colors and hatch patterns for each box
    for box, color, hatch in zip(bp['boxes'], colors, hatch_patterns):
        box.set(facecolor=color, hatch=hatch)
    
    # Set colors for whiskers and outliers
    for whisker in bp['whiskers']:
        whisker.set(color='black')
    
    for cap in bp['caps']:
        cap.set(color='black')
    
    for median in bp['medians']:
        median.set(color='black')
    
    for flier in bp['fliers']:
        flier.set(marker='o', color='black', alpha=0.5)
    
    plt.ylabel('Segment bytes, MB')
    plt.title('Segment Size Comparison (5-95 Percentile)')
    plt.tight_layout()
    plt.savefig('size_boxplot.png', dpi=300)
    plt.show()
def plot_total_size(baseline_df, your_method_df):
    # Plot the total size of the video streamed for the baseline and your method
    # X-axis: method, Y-axis: total size in MB
    plt.figure(figsize=(10, 6))
    # set bar in different color
    plt.bar(['Baseline', 'Your Method'], [baseline_df['BYTES'].sum() / 1e6, your_method_df['BYTES'].sum() / 1e6], color=['#EB5E55', '#8F4899'], width=0.5)
    plt.ylabel('Total Size (MB)')
    plt.title('Total Size Comparison')
    plt.savefig('total_size_comparison.png')
    plt.show()

if __name__ == '__main__':
    # Read the CSV files
    basline_csv_path = './Training_output/segue_wide_eye/trace_0.txt.csv'
    my_method_csv_path = './Training_output/QAOCS/bigbuckbunny360p24trace_0.txt_test_simulation.csv'
    my_method_df = pd.read_csv(my_method_csv_path)
    baseline_df = pd.read_csv(basline_csv_path)
    # baseline time scale is seconds, convert to milliseconds including the  buffer state, and segment progressive and rebuff
    baseline_df['DURATION'] = baseline_df['DURATION'] * 1000
    baseline_df['BUFFER_STATE'] = baseline_df['BUFFER_STATE'] * 1000
    baseline_df['REBUF'] = baseline_df['REBUF'] * 1000

    # baseline VMAF is normalize by Duration/4 we need to convert it to the original scale
    baseline_df['VMAF'] = baseline_df['VMAF'] * 4 / (baseline_df['DURATION']/1000)



    plot_segment_duration(baseline_df, my_method_df)
    plot_segment_size(baseline_df, my_method_df)
    plot_buffer_state(baseline_df, my_method_df)
    plot_vmaf_score(baseline_df, my_method_df)
    plot_bitrate(baseline_df, my_method_df)
    plot_size_boxplot(baseline_df, my_method_df)
    plot_total_size(baseline_df, my_method_df)
    #plot_subplots(baseline_df, my_method_df)

 
    
    # Calculate average metrics for your method
    our_method_avg_duration = my_method_df['DURATION'].mean()
    our_method_avg_size = my_method_df['BYTES'].mean()
    our_method_avg_buffer_state = my_method_df['BUFFER_STATE'].mean()
    our_method_avg_vmaf = my_method_df['VMAF'].mean()
    our_method_avg_bitrate = my_method_df['BITRATE'].mean()
    our_method_total_size = my_method_df['BYTES'].sum()

    # Calculate average metrics for the baseline
    baseline_avg_duration = baseline_df['DURATION'].mean()
    baseline_avg_size = baseline_df['BYTES'].mean()
    baseline_avg_buffer_state = baseline_df['BUFFER_STATE'].mean()
    baseline_avg_vmaf = baseline_df['VMAF'].mean()
    baseline_avg_bitrate = baseline_df['BITRATE'].mean()
    baseline_avg_duration = baseline_df['DURATION'].mean()

    print("Our Method:")
    print(f'Average Duration: {our_method_avg_duration/1000:.2f}s')
    print(f'Average Size: {our_method_avg_size/1e6:.2f}MB')
    print(f'Average Buffer State: {our_method_avg_buffer_state/1000:.2f}s')
    print(f'Average VMAF Score:" {our_method_avg_vmaf}')
    print(f'Average Bitrate: {our_method_avg_bitrate}bps')
    print(f'Total Size: {our_method_total_size/1e6:.2f}MB')

    print("\nBaseline:")
    print(f'Average Duration: {baseline_avg_duration/1000:.2f}s')
    print(f'Average Size: {baseline_avg_size/1e6:.2f}MB')
    print(f'Average Buffer State: {baseline_avg_buffer_state/1000:.2f}s')
    print(f'Average VMAF Score: {baseline_avg_vmaf}')
    print(f'Average Bitrate: {baseline_avg_bitrate}bps')
    print(f'Total Size: {baseline_df["BYTES"].sum()/1e6:.2f}MB')

