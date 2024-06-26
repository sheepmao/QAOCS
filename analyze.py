import pandas as pd
import matplotlib.pyplot as plt
import os
import csv


def pmkdir(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)

def plot_segment_duration(baseline_df, your_method_df,save_path):
    # Plot the segment duration for the baseline and your method in bar chart
    # X-axis: segment number, Y-axis: segment duration
    plt.figure(figsize=(10, 6))
    plt.bar(your_method_df['SEGMENT_NO'], your_method_df['DURATION']/1000, label='Your Method')
    plt.bar(baseline_df['SEGMENT_PROGRESSIVE'], baseline_df['DURATION']/1000, label='Baseline')
    plt.xlabel('Segment Number')
    plt.ylabel('Duration (seconds)')
    plt.title('Segment Duration Comparison')
    plt.legend()
    plt.savefig(save_path+'segment_duration_comparison.png')
    plt.show()
    plt.close()
def plot_segment_size(baseline_df, your_method_df,save_path):
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
    plt.savefig(save_path+'segment_size_comparison.png')
    plt.show()
    plt.close()

def plot_buffer_state(baseline_df, your_method_df,save_path):
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
    plt.savefig(save_path+'buffer_state_comparison.png')
    plt.show()
    plt.close()
def plot_vmaf_score(baseline_df, your_method_df,save_path):
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
    plt.savefig(save_path+'vmaf_score_comparison.png')
    plt.show()
    plt.close()
def plot_bitrate(baseline_df, your_method_df,save_path):
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
    plt.savefig(save_path+'bitrate_comparison.png')
    plt.show()
    plt.close()
def plot_size_boxplot(baseline_df, your_method_df,save_path):
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
    plt.savefig(save_path+'size_boxplot.png', dpi=300)
    plt.show()
    plt.close()
def plot_total_size(baseline_df, your_method_df,save_path):
    # Plot the total size of the video streamed for the baseline and your method
    # X-axis: method, Y-axis: total size in MB
    plt.figure(figsize=(10, 6))
    # set bar in different color
    plt.bar(['Baseline', 'Your Method'], [baseline_df['BYTES'].sum() / 1e6, your_method_df['BYTES'].sum() / 1e6], color=['#EB5E55', '#8F4899'], width=0.5)
    plt.ylabel('Total Size (MB)')
    plt.title('Total Size Comparison')
    plt.savefig(save_path+'total_size_comparison.png')
    plt.show()
    plt.close()


def Calculate_average_metrics(df):
    Switch_count = (df["QUALITY_INDEX"]-df["QUALITY_INDEX"].shift(1).fillna(0)!=0).sum()
    avg_duration = df['DURATION'].mean()
    avg_size = df['BYTES'].mean()
    avg_buffer_state = df['BUFFER_STATE'].mean()
    avg_vmaf = (df['VMAF'].mean()*df['DURATION']/1000).sum()/ (df['DURATION']/1000).sum()
    avg_bitrate = df['BITRATE'].mean()
    #stall_ratio = df["REBUF"].sum()/df["DURATION"].sum()*100
    # count the number of rebuffering events / total segment number
    stall_ratio = (df["REBUF"]>0).sum()/len(df)*100
    stall_time = df["REBUF"].sum()
    switch_ratio = Switch_count/len(df)
    total_size = df['BYTES'].sum()
    #vmaf smoothness = average of difference between consecutive vmaf scores
    avg_vmaf_smoothness = (df['VMAF'].diff().abs().sum()/(len(df)-1))
    print(f'Average Duration: {avg_duration/1000:.2f}s')
    print(f'Average Size: {avg_size/1e6:.2f}MB')
    print(f'Average Buffer State: {avg_buffer_state/1000:.2f}s')
    print(f'Average VMAF Score:" {avg_vmaf}')
    print(f'Average VMAF Smoothness: {avg_vmaf_smoothness:.2f}')
    print(f'Average Bitrate: {avg_bitrate}bps')
    print(f'Stall Ratio: {stall_ratio:.2f}%')
    print(f'Stall Time: {stall_time/1000:.2f}s')
    print(f'Switch Ratio: {switch_ratio:.2f}%')
    print(f'Total Size: {total_size/1e6:.2f}MB')

    return {
    'Average Duration(s)': f'{avg_duration/1000:.2f}',
    'Average Size(MB)': f'{avg_size/1e6:.2f}',
    'Average Buffer State(s)': f'{avg_buffer_state/1000:.2f}',
    'Average VMAF Score': avg_vmaf,
    'Average VMAF Smoothness': f'{avg_vmaf_smoothness:.2f}',
    'Average Bitrate(bps)': f'{avg_bitrate}',
    'Stall Ratio(%)': f'{stall_ratio:.2f}',
    'Stall Time(s)': f'{stall_time/1000:.2f}',
    'Switch Ratio(%)': f'{switch_ratio:.2f}',
    'Total Size(MB)': f'{total_size/1e6:.2f}'
        }
if __name__ == '__main__':
    # Read the CSV files
    video_list = ['BBB_360p24','LOL_3D','sport_highlight','sport_long_take','TOS_360p24','video_game','underwater']
    for video_name in video_list:
    #video_name = 'video_game' # 'BBB' or 'TOS' 
        wide_eye_csv_path = './Training_output/segue_wide_eye/{}/'.format(video_name)
        constant_csv_path_ = './Training_output/segue_constant_5/{}/'.format(video_name)
        gop_5_csv_path_ = './Training_output/segue_GOP-5/{}/'.format(video_name)
        our_PPO_csv_path = './Training_output/QAOCS/{}/'.format(video_name)

        results=[]
        for file in os.listdir(wide_eye_csv_path):
            if file.endswith(".csv"):
                baseline_csv_path = os.path.join(wide_eye_csv_path, file)
                my_method_csv_path = os.path.join(our_PPO_csv_path, file)
                gop_5_csv_path = os.path.join(gop_5_csv_path_, file)
                constant_csv_path = os.path.join(constant_csv_path_, file)
            


                save_path = './Figures/'+video_name+'/'
                pmkdir(save_path)

                # read the csv files
                my_method_df = pd.read_csv(my_method_csv_path)
                baseline_df = pd.read_csv(baseline_csv_path)
                gop_5_df = pd.read_csv(gop_5_csv_path)
                constant_5_df = pd.read_csv(constant_csv_path)

                # segue framwork time scale is seconds, convert to milliseconds including the  buffer state, and segment progressive and rebuff
                baseline_df['DURATION'] = baseline_df['DURATION'] * 1000
                baseline_df['BUFFER_STATE'] = baseline_df['BUFFER_STATE'] * 1000
                baseline_df['REBUF'] = baseline_df['REBUF'] * 1000
                # baseline VMAF is normalize by Duration/4 we need to convert it to the original scale
                baseline_df['VMAF'] = baseline_df['VMAF'] * 4 / (baseline_df['DURATION']/1000)

                gop_5_df['DURATION'] = gop_5_df['DURATION'] * 1000
                gop_5_df['BUFFER_STATE'] = gop_5_df['BUFFER_STATE'] * 1000
                gop_5_df['REBUF'] = gop_5_df['REBUF'] * 1000
                gop_5_df['VMAF'] = gop_5_df['VMAF'] * 4 / (gop_5_df['DURATION']/1000)

                constant_5_df['DURATION'] = constant_5_df['DURATION'] * 1000
                constant_5_df['BUFFER_STATE'] = constant_5_df['BUFFER_STATE'] * 1000
                constant_5_df['REBUF'] = constant_5_df['REBUF'] * 1000
                constant_5_df['VMAF'] = constant_5_df['VMAF'] * 4 / (constant_5_df['DURATION']/1000)




                plot_segment_duration(baseline_df, my_method_df,save_path)
                plot_segment_size(baseline_df, my_method_df,save_path)
                plot_buffer_state(baseline_df, my_method_df,save_path)
                plot_vmaf_score(baseline_df, my_method_df,save_path)
                plot_bitrate(baseline_df, my_method_df,save_path)
                plot_size_boxplot(baseline_df, my_method_df,save_path)
                plot_total_size(baseline_df, my_method_df,save_path)
                #plot_subplots(baseline_df, my_method_df)



        
                # Calculate average metrics for your method
                # Switch ratio is the number of switches between different quality levels
                our_method_switch_count = (my_method_df["QUALITY_INDEX"]-my_method_df["QUALITY_INDEX"].shift(1).fillna(0)!=0).sum()
                our_method_avg_duration = my_method_df['DURATION'].mean()
                our_method_avg_size = my_method_df['BYTES'].mean()
                our_method_avg_buffer_state = my_method_df['BUFFER_STATE'].mean()
                our_method_avg_vmaf = (my_method_df['VMAF'].mean()*my_method_df['DURATION']/1000).sum()/ (my_method_df['DURATION']/1000).sum()
                our_method_avg_bitrate = my_method_df['BITRATE'].mean()
                our_method_stall_ratio = my_method_df["REBUF"].sum()/my_method_df["DURATION"].sum()*100
                our_method_switch_ratio = our_method_switch_count/len(my_method_df)
                our_method_total_size = my_method_df['BYTES'].sum()

                # Calculate average metrics for the baseline
                baseline_switch_count = (baseline_df["QUALITY_INDEX"]-baseline_df["QUALITY_INDEX"].shift(1).fillna(0)!=0).sum()
                baseline_avg_duration = baseline_df['DURATION'].mean()
                baseline_avg_size = baseline_df['BYTES'].mean()
                baseline_avg_buffer_state = baseline_df['BUFFER_STATE'].mean()
                baseline_avg_vmaf = (baseline_df['VMAF']*baseline_df['DURATION']/1000).sum()/ (baseline_df['DURATION']/1000).sum()
                baseline_avg_bitrate = baseline_df['BITRATE'].mean()
                baseline_stall_ratio = baseline_df["REBUF"].sum()/baseline_df["DURATION"].sum()*100
                baseline_switch_ratio = baseline_switch_count/len(baseline_df)
                baseline_avg_duration = baseline_df['DURATION'].mean()

                # Calculate average metrics for the GOP-5
                gop_5_switch_count = (gop_5_df["QUALITY_INDEX"]-gop_5_df["QUALITY_INDEX"].shift(1).fillna(0)!=0).sum()
                gop_5_avg_duration = gop_5_df['DURATION'].mean()
                gop_5_avg_size = gop_5_df['BYTES'].mean()
                gop_5_avg_buffer_state = gop_5_df['BUFFER_STATE'].mean()
                gop_5_avg_vmaf = (gop_5_df['VMAF']*gop_5_df['DURATION']/1000).sum()/ (gop_5_df['DURATION']/1000).sum()
                gop_5_avg_bitrate = gop_5_df['BITRATE'].mean()
                gop_5_stall_ratio = gop_5_df["REBUF"].sum()/gop_5_df["DURATION"].sum()*100
                gop_5_switch_ratio = gop_5_switch_count/len(gop_5_df)
                gop_5_total_size = gop_5_df['BYTES'].sum()

                # Calculate average metrics for the constant-5
                constant_5_switch_count = (constant_5_df["QUALITY_INDEX"]-constant_5_df["QUALITY_INDEX"].shift(1).fillna(0)!=0).sum()
                constant_5_avg_duration = constant_5_df['DURATION'].mean()
                constant_5_avg_size = constant_5_df['BYTES'].mean()
                constant_5_avg_buffer_state = constant_5_df['BUFFER_STATE'].mean()
                constant_5_avg_vmaf = (constant_5_df['VMAF']*constant_5_df['DURATION']/1000).sum()/ (constant_5_df['DURATION']/1000).sum()
                constant_5_avg_bitrate = constant_5_df['BITRATE'].mean()
                constant_5_stall_ratio = constant_5_df["REBUF"].sum()/constant_5_df["DURATION"].sum()*100
                constant_5_switch_ratio = constant_5_switch_count/len(constant_5_df)
                constant_5_total_size = constant_5_df['BYTES'].sum()

                print("Our Method:")
                our_method_metrics = Calculate_average_metrics(my_method_df)
        

                print("\nBaseline:")
                baseline_metrics = Calculate_average_metrics(baseline_df)

                print("\nGOP-5:")
                gop_5_metrics  = Calculate_average_metrics(gop_5_df)

                print("\nConstant-5:")
                constant_5_metrics = Calculate_average_metrics(constant_5_df)
                            # Add the results to the list
                results.append({
                    'File': file,
                    'QAOCS': our_method_metrics,
                    'Segue': baseline_metrics,
                    'GOP-5': gop_5_metrics,
                    'Constant-5': constant_5_metrics
                })
        # Write the results to a CSV file
        output_file = video_name + '_performance_comparison.csv'
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['File', 'Method', 'Average Duration(s)', 'Average Size(MB)', 'Average Buffer State(s)',
                        'Average VMAF Score','Average VMAF Smoothness', 'Average Bitrate(bps)', 'Stall Time(s)','Stall Ratio(%)', 'Switch Ratio(%)', 'Total Size(MB)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                file = result['File']
                for method, metrics in result.items():
                    if method != 'File':
                        row = {'File': file, 'Method': method, **metrics}
                        writer.writerow(row)

        print(f"{video_name} Performance comparison results saved to {video_name+output_file}")
