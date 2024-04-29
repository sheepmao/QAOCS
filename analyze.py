import pandas as pd
import matplotlib.pyplot as plt


def plot_segment_duration(baseline_df, your_method_df):
    plt.figure(figsize=(10, 6))
    plt.plot(your_method_df['SEGMENT_NO'], your_method_df['DURATION'], label='Your Method')
    plt.plot(baseline_df['SEGMENT_PROGRESSIVE'], baseline_df['DURATION'], label='Baseline')
    plt.xlabel('Segment Number')
    plt.ylabel('Duration (Mileseconds)')
    plt.title('Segment Duration Comparison')
    plt.legend()
    plt.savefig('segment_duration_comparison.png')
    plt.show()
def plot_segment_size(baseline_df, your_method_df):
    plt.figure(figsize=(10, 6))
    plt.plot(your_method_df['SEGMENT_NO'], your_method_df['BYTES'], label='Your Method')
    plt.plot(baseline_df['SEGMENT_PROGRESSIVE'], baseline_df['BYTES'], label='Baseline')
    plt.xlabel('Segment Number')
    plt.ylabel('Size (bytes)')
    plt.title('Segment Size Comparison')
    plt.legend()
    plt.savefig('segment_size_comparison.png')
    plt.show()

def plot_buffer_state(baseline_df, your_method_df):
    plt.figure(figsize=(10, 6))
    plt.plot(your_method_df['SEGMENT_NO'], your_method_df['BUFFER_STATE'], label='Your Method')
    plt.plot(baseline_df['SEGMENT_PROGRESSIVE'], baseline_df['BUFFER_STATE'], label='Baseline')
    plt.xlabel('Segment Number')
    plt.ylabel('Buffer State (seconds)')
    plt.title('Buffer State Comparison')
    plt.legend()
    plt.savefig('buffer_state_comparison.png')
    plt.show()
def plot_vmaf_score(baseline_df, your_method_df):
    plt.figure(figsize=(10, 6))
    plt.plot(your_method_df['SEGMENT_NO'], your_method_df['VMAF'], label='Your Method')
    plt.plot(baseline_df['SEGMENT_PROGRESSIVE'], baseline_df['VMAF'], label='Baseline')
    plt.xlabel('Segment Number')
    plt.ylabel('VMAF Score')
    plt.title('Quality (VMAF) Comparison')
    plt.legend()
    plt.savefig('vmaf_score_comparison.png')
    plt.show()
def plot_bitrate(baseline_df, your_method_df):
    plt.figure(figsize=(10, 6))
    plt.plot(your_method_df['SEGMENT_NO'], your_method_df['BITRATE'], label='Your Method')
    plt.plot(baseline_df['SEGMENT_PROGRESSIVE'], baseline_df['BITRATE'], label='Baseline')
    plt.xlabel('Segment Number')
    plt.ylabel('Bitrate (bps)')
    plt.title('Bitrate Comparison')
    plt.legend()
    plt.savefig('bitrate_comparison.png')
    plt.show()

def calculate_stall_ratio(data):
    # Calculate the stall ratio
    # stall event count = where rebuf > 0
    stall_event_count = sum(data['REBUF'] > 0)
    # stall ration = stall event count / total number of segments
    stall_ratio = stall_event_count / len(data)
    return stall_ratio


def plot_subplots(baseline_df, your_method_df):

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    # vmaf vs stall ratio from the baseline and your method
    vmaf_vs_stall = {
        'Baseline': [(x, y) for x, y in zip(baseline_df['VMAF'], baseline_df['STALL_RATIO'])],
        'Your Method': [(x, y) for x, y in zip(your_method_df['VMAF'], your_method_df['STALL_RATIO'])]
    }
    vmaf_vs_vmaf_change = {
        'Baseline': [(x, y) for x, y in zip(baseline_df['VMAF'], baseline_df['VMAF_CHANGE'])],
        'Your Method': [(x, y) for x, y in zip(your_method_df['VMAF'], your_method_df['VMAF_CHANGE'])]
    }
    qoe_dnn_vs_buffer = {
        'Baseline': [(x, y) for x, y in zip(baseline_df['REWARD'], baseline_df['BUFFER_STATE'])],
        'Your Method': [(x, y) for x, y in zip(your_method_df['REWARD'], your_method_df['BUFFER_STATE'])]
    }
    qoe_dnn_values = sorted(list(set(baseline_df['REWARD'] + your_method_df['REWARD'])))
    cdf_qoe_dnn = {
        'Baseline': [sum(baseline_df['REWARD'] <= x) / len(baseline_df) for x in qoe_dnn_values],
        'Your Method': [sum(your_method_df['REWARD'] <= x) / len(your_method_df) for x in qoe_dnn_values]
    }
    # calculate the stall ratio and buffer state from data
    stall_ratio = list(set(baseline_df['STALL_RATIO']))
    buffer_sizes = list(set(baseline_df['BUFFER_STATE']))
    # Plot (a) VMAF vs. Stall Ratio
    for algo, vmaf_values in vmaf_vs_stall.items():
        axs[0, 0].plot([x[0] for x in vmaf_values], [y for y in stall_ratio], marker='o', label=algo)
        axs[0, 0].boxplot([x[1]-x[0] for x in vmaf_values], positions=stall_ratio, widths=0.1, vert=False, sym='')
    axs[0, 0].set_xlabel('Video Quality (VMAF)')
    axs[0, 0].set_ylabel('Time Spent on Stall (%)')
    axs[0, 0].set_title('(a) VMAF vs. Stall Ratio')
    axs[0, 0].legend()

    # # Plot (b) VMAF vs. VMAF Change
    # for algo, vmaf_values in vmaf_vs_vmaf_change.items():
    #     axs[0, 1].plot([x[0] for x in vmaf_values], quality_smoothness, marker='o', label=algo)
    #     axs[0, 1].boxplot([x[1]-x[0] for x in vmaf_values], positions=quality_smoothness, widths=0.5, vert=True, sym='')
    # axs[0, 1].set_xlabel('Video Quality (VMAF)')
    # axs[0, 1].set_ylabel('Quality Smoothness (VMAF)')
    # axs[0, 1].set_title('(b) VMAF vs. VMAF Change')
    # axs[0, 1].legend()

    # Plot (c) QoE_DNN vs. Buffer
    for algo, qoe_values in qoe_dnn_vs_buffer.items():
        axs[1, 0].plot([x[0] for x in qoe_values], buffer_sizes, marker='o', label=algo)
        axs[1, 0].boxplot([x[1]-x[0] for x in qoe_values], positions=buffer_sizes, widths=1, vert=True, sym='')
    axs[1, 0].set_xlabel('QoE')
    axs[1, 0].set_ylabel('Buffer (s)')
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

    # Save the plot
    plt.savefig('subplots.png')
    # Display the plot
    plt.show()


if __name__ == '__main__':
    # Read the CSV files
    basline_csv_path = './Training_output/segue_wide_eye/trace_96.txt.csv'
    my_method_csv_path = './Training_output/sheepmao1/Simulation_result/bigbuckbunny360p24eposide_480simulation.csv'
    your_method_df = pd.read_csv(my_method_csv_path)
    baseline_df = pd.read_csv(basline_csv_path)
    # baseline time scale is seconds, convert to milliseconds including the  buffer state, and segment progressive and rebuff
    baseline_df['DURATION'] = baseline_df['DURATION'] * 1000
    baseline_df['BUFFER_STATE'] = baseline_df['BUFFER_STATE'] * 1000
    baseline_df['REBUF'] = baseline_df['REBUF'] * 1000

    # Calculate the stall ratio for the baseline and your method, and add it to the dataframes
    baseline_df['STALL_RATIO'] = calculate_stall_ratio(baseline_df)
    your_method_df['STALL_RATIO'] = calculate_stall_ratio(your_method_df)


    plot_segment_duration(baseline_df, your_method_df)
    plot_segment_size(baseline_df, your_method_df)
    plot_buffer_state(baseline_df, your_method_df)
    plot_vmaf_score(baseline_df, your_method_df)
    plot_bitrate(baseline_df, your_method_df)
    plot_subplots(baseline_df, your_method_df)

    # Calculate average metrics for your method
    your_method_avg_duration = your_method_df['DURATION'].mean()
    your_method_avg_size = your_method_df['BYTES'].mean()
    your_method_avg_buffer_state = your_method_df['BUFFER_STATE'].mean()
    your_method_avg_vmaf = your_method_df['VMAF'].mean()
    your_method_avg_bitrate = your_method_df['BITRATE'].mean()

    # Calculate average metrics for the baseline
    baseline_avg_duration = baseline_df['DURATION'].mean()
    baseline_avg_size = baseline_df['BYTES'].mean()
    baseline_avg_buffer_state = baseline_df['BUFFER_STATE'].mean()
    baseline_avg_vmaf = baseline_df['VMAF'].mean()
    baseline_avg_bitrate = baseline_df['BITRATE'].mean()

    print("Your Method:")
    print("Average Duration:", your_method_avg_duration)
    print("Average Size:", your_method_avg_size)
    print("Average Buffer State:", your_method_avg_buffer_state)
    print("Average VMAF Score:", your_method_avg_vmaf)
    print("Average Bitrate:", your_method_avg_bitrate)

    print("\nBaseline:")
    print("Average Duration:", baseline_avg_duration)
    print("Average Size:", baseline_avg_size)
    print("Average Buffer State:", baseline_avg_buffer_state)
    print("Average VMAF Score:", baseline_avg_vmaf)
    print("Average Bitrate:", baseline_avg_bitrate)