import numpy as np
import os
from ffmpeg_quality_metrics import FfmpegQualityMetrics, VmafOptions
import abr as ABR
import pandas as pd
# For video features
import cv2
from skimage.feature import graycomatrix, graycoprops
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import PIL.Image as Image
S_INFO = 10
S_LEN = 10  # take how many frames in the past
A_DIM = 2
REF_DUR = 4.0  # research shows 4 is a good value for segment duration

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42
CRF_LEVELS = 51
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
NOISE_LOW = 0.9
NOISE_HIGH = 1.1

#VMAF_HDTV_HEIGHT = '1080' VMAF_HDTV_WIDTH = '1920'
VMAF_HDTV_MODEL = './vmaf/model/vmaf_float_v0.6.1.pkl'



RESOLUTION_LIST = [(640, 360), (854, 480), (1280, 720), (1920, 1080)]

def pmkdir(kdir):
    if not os.path.exists(kdir):
        os.makedirs(kdir)


class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, video, random_seed=RANDOM_SEED,writer=None):
        assert len(all_cooked_time) == len(all_cooked_bw) , "Mismatched cooked time and bandwidth arrays"
        self.writer = writer

        np.random.seed(random_seed)
        self.alpha = 0.1  # for vmaf(0-100)
        self.beta = -0.3  # for smoothness(abs(last_vmaf - vmaf)-> 0-100)
        self.gamma = -5.0  # for stall probability(0-1)
        self.delta = -0.1   # for log bitrate(1-10)
        self.epsilon = 0.2 # for ref_dur_ratio 
        self.eposide = 0
        self.step_count = 0


        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw
        self.video = video
        self.total_video_time = video.load_duration() * MILLISECONDS_IN_SECOND
        self.fps = video.load_fps()

        self.video_chunk_counter = 0
        self.buffer_size = 0
        self.video_time = 0
        self.arrival_time = []
        self.service_time = []
        self.vmaf = [0.]

        # store the list of each video chunk information
        self.B_list = []
        self.CRF = []

        
        # pick a random trace file
        self.trace_idx = np.random.randint(len(self.all_cooked_time)) # random select a trace file
        self.cooked_time = self.all_cooked_time[self.trace_idx]  # Time
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]  # Bandwidth

        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))  # trace file pointer
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]  # last time download

        self.video_size = []  # in bytes
        self._vmaf_data = []

        self.data = {
        "TIME": [],
        "SEGMENT_NO"	: [],
        "DURATION":   [],
        "BYTES":   [],
        "QUALITY_INDEX"	:   [],
        "RESOLUTION"	:   [],
        "BITRATE"	:   [],
        "VMAF"	:   [],
        "REBUF"	:   [],
        "BUFFER_STATE"	:   [],
        "DELAY"	:   [],
        "REWARD": []
        }
    def reset(self):
            # reset the environment
            self.buffer_size = 0
            self.video_chunk_counter = 0
            self.video_time = 0
            self.arrival_time = []
            self.service_time = []
            self.video_size = []
            self.B_list = []
            self.CRF = []

            # pick a random trace file
            self.trace_idx = np.random.randint(len(self.all_cooked_time))
            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            # randomize the start point of the video
            # note: trace file starts with time 0
            self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]
            # reset the data for the next video
            self.data = {
            "TIME": [],
            "SEGMENT_NO"	: [],
            "DURATION":   [],
            "BYTES":   [],
            "QUALITY_INDEX"	:   [],
            "RESOLUTION"	:   [],
            "BITRATE"	:   [],
            "VMAF"	:   [],
            "REBUF"	:   [],
            "BUFFER_STATE"	:   [],
            "DELAY"	:   [],
            "REWARD": []
            }
    def get_video_chunk(self, B, CRF):

        assert CRF >= 0
        assert CRF <= CRF_LEVELS, print('wrong set CRF as: ', CRF)

        start_time = self.video_time / 1000
        end_time = (self.video_time + B) / 1000 if self.video_time + B < self.total_video_time \
                                                else self.total_video_time / 1000
        # Crop the video to the specific time range obtained from the agent decision and current video time
        #save_path = self.video.video_name().split('.mp4')[0] +str(self.video_chunk_counter)+'_reference_video.mp4'
        save_path = self.video.video_name().split('.mp4')[0] + '_reference_video.mp4'
        video_chunk = self.video.crop_video(start_time, end_time, save_path)
        # Rescale the video to the specific quality level obtained from the agent decision
        # Get the corresponding video size and record the segment duration
        ## 通过CRF和video_chunk计算视频大小 和 视频 named by video chunk number 
        #b, video_rescale_quality = self.culculate_bits(video_chunk, CRF,self.video_chunk_counter)

        # crop the video to different resolution
        ## return the video size and the video object list order by resolution low to high
        bitrates,sizes, rescaled_videos = self.culculate_bits_multi(video_chunk, CRF, self.video_chunk_counter)
        
        # ABR to select which resoltion and corresponding video size
        abr = ABR.Abr()
        quality_level = abr.abr(self, bitrates)
        self.data["QUALITY_INDEX"].append(quality_level)

        b = sizes[quality_level]
        self.data["RESOLUTION"].append(RESOLUTION_LIST[quality_level])
        self.data["BITRATE"].append(bitrates[quality_level])
        self.data["BYTES"].append(b)
        self.data["DURATION"].append(B)
        self.data["SEGMENT_NO"].append(self.video_chunk_counter)
        self.data["TIME"].append(self.video_time)

        video_rescale_quality = rescaled_videos[quality_level]
        self.video_size.append(b)
        self.service_time.append(B)  # 服务时间/ms

        video_chunk_size = self.video_size[self.video_chunk_counter]

        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes

        while True:  # download video chunk over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr] * B_IN_MB / BITS_IN_BYTE  # 下载速度MB/s
            duration = self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time  # 持續時間s

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION  # 在時間內能下載多少MB的packet

            # 如果能下載的量+先前下載的量大於video_chunk_size(video_chunk_counter_sent是之前下载的量)
            if video_chunk_counter_sent + packet_payload > video_chunk_size:
                # fractional_time 是當前duration內的下載時間，delay是下載當前packet的總時間
                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                  throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                assert (self.last_mahimahi_time <= self.cooked_time[self.mahimahi_ptr])
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0

        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT
        self.data["DELAY"].append(delay)
        # add a multiplicative noise to the delay
        delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)
        self.arrival_time.append(delay)  # 到達時間/ms

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        self.data["REBUF"].append(rebuf)
        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)
        # add in the new chunk
        self.buffer_size += B
        
        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > BUFFER_THRESH:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - BUFFER_THRESH
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                         DRAIN_BUFFER_SLEEP_TIME
            self.buffer_size -= sleep_time

            while True:
                duration = self.cooked_time[self.mahimahi_ptr] \
                           - self.last_mahimahi_time
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0
        self.data["BUFFER_STATE"].append(self.buffer_size)
        buffer_levels = self.calculate_buffer_levels()
        pst = self.calculate_stall_probability(buffer_levels)
        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size

        self.video_chunk_counter += 1
        # 计算vmaf分数
        self.vmaf.append(
            self.calculate_vmaf(video_chunk,
                                video_rescale_quality,
                                )
        )
        self.data["VMAF"].append(self.vmaf[self.video_chunk_counter])

        if self.video_chunk_counter == 1:
            last_vmaf = self.vmaf[self.video_chunk_counter]
        else:
            last_vmaf = self.vmaf[self.video_chunk_counter - 1]
        vmaf = self.vmaf[self.video_chunk_counter]

        self.video_time += B
        video_time_remain = self.total_video_time - self.video_time

        ref_dur_ration = REF_DUR*1000  /B
        # Calculate the reward
        #VMAF at least 70 is better
        QoE = self.alpha * (vmaf-70) + self.beta * abs(last_vmaf - vmaf) + self.gamma * pst\
              + self.delta * np.log(bitrates[quality_level]) + self.epsilon * (1 - ref_dur_ration)
        # print the reward separately to check the value
        print('QoE:', QoE, 'vmaf:', vmaf, 'smoothness:', abs(last_vmaf - vmaf), 'pst:', pst, 'log bitrate:', np.log(bitrates), 'ref_dur_ratio:', ref_dur_ration)
        #QoE = self.alpha * vmaf + self.beta * abs(last_vmaf - vmaf) + self.gamma * pst
        reward = QoE
        self.data["REWARD"].append(reward)


        self.writer.add_scalar('VMAF', vmaf, self.step_count)
        self.writer.add_scalar('PST', pst, self.step_count)
        self.writer.add_scalar('Video Time', self.video_time, self.step_count)
        self.writer.add_scalar('Buffer Size', self.buffer_size, self.step_count)
        self.writer.add_scalar('Smoothness', abs(last_vmaf - vmaf), self.step_count)
        self.writer.add_scalar('Log Bitrates', np.log(bitrates[quality_level]), self.step_count)
        self.writer.add_scalar('CRF', CRF, self.step_count)
        self.writer.add_scalar('Duration', B, self.step_count)
        self.writer.add_scalars('QoE Factors', {'VMAF':vmaf,'PST':pst,'Video Time':self.video_time,\
                                               'Buffer Size':self.buffer_size,
                                               'Smoothness':abs(last_vmaf-vmaf),
                                               'Log Bitrates':np.log(bitrates[quality_level])},
                                                 self.step_count)
        self.step_count += 1
        # check whether the video is finished
        end_of_video = False
        if video_time_remain <= 500:  # remain video < 500ms -> end of video
            # Set the end flag
            end_of_video = True
            # store the information into csv before the reset
            #trace_name = 'trace_' + str(self.trace_idx)
            csv_file_save_path = 'Videos_result/' + self.video.video_name().split('.mp4')[0] \
                + 'eposide_'+str(self.eposide) +'simulation.csv'
            self.eposide += 1
            data_frame = pd.DataFrame(self.data)
            data_frame.to_csv(csv_file_save_path, index=False)
            self.reset()

        return delay, sleep_time, return_buffer_size / MILLISECONDS_IN_SECOND, \
               rebuf / MILLISECONDS_IN_SECOND, video_chunk_size, end_of_video, pst, vmaf, last_vmaf, reward

    def culculate_bits(self, video, CRF,video_chunk_counter):
        # 计算bits
        # create a directory to store the video chunks if it does not exist
        Videos_dir = 'Videos_result'
        if not os.path.exists(Videos_dir):
            pmkdir(Videos_dir)
        Video_dir = 'Videos_result/' + video.video_name().replace(
            '_'+str(self.video_chunk_counter)+'_reference_video.mp4','')
        if not os.path.exists(Video_dir):
            pmkdir(Video_dir)
        print("video_name:", video.video_name())
        # store each video chunk in a file
        save_path = Video_dir + '/'+str(video_chunk_counter) + '.mp4'
        print("chunk_path",save_path)
        v = video.rescale_h264_constant_quality(save_path, CRF, 1920, 1080, 0, force=True)
        return v.load_bytes(), v
    def culculate_bits_multi(self, video, CRF, video_chunk_counter):
        Videos_dir = 'Videos_result'
        if not os.path.exists(Videos_dir):
            pmkdir(Videos_dir)
        Video_dir = 'Videos_result/' + video.video_name().replace('_reference_video.mp4','')
        if not os.path.exists(Video_dir):
            pmkdir(Video_dir)
        # 计算bits for different resolution
        Resolution = RESOLUTION_LIST
        videos = []
        videos_sizes = []
        bitrates = []
        # store each video chunk in a file for different resolution named by video chunk number and resolution
        for i in range(len(Resolution)):
            #create directories  for each resolution
            resolution_dir = Video_dir + '/' + str(Resolution[i][0]) + 'x' + str(Resolution[i][1])
            if not os.path.exists(resolution_dir):
                pmkdir(resolution_dir)
            # store each video chunk in a file
            save_path = resolution_dir + '/'+str(video_chunk_counter) + '.mp4'
            v = video.rescale_h264_constant_quality(save_path, CRF, Resolution[i][0], Resolution[i][1], 0, force=True)
            videos.append(v)
            videos_sizes.append(v.load_bytes())
            bitrates.append(v.load_bitrate())
        return bitrates,videos_sizes, videos

    def calculate_vmaf(self, reference_video, video):
        # 根据video，CRF，计算视频segment的average vmaf
        ffqm = FfmpegQualityMetrics(reference_video.video_path(), video.video_path())
        metrics = ffqm.calculate(
            ["vmaf"],
            VmafOptions(
                model_path='./vmaf/model/vmaf_float_v0.6.1.json'
            ),
        )
        vmaf_sum = 0.
        vmaf_len = 0.
        for metric in metrics['vmaf']:
            vmaf_sum += metric['vmaf']
            vmaf_len += 1

        return vmaf_sum / vmaf_len

    def calculate_buffer_levels(self, initial_buffer=20000):
        buffer_levels = [initial_buffer]
        for arrival, service in zip(self.arrival_time, self.service_time):
            # New buffer level without the restriction of not going below zero
            new_level = buffer_levels[-1] - arrival + service
            buffer_levels.append(min(new_level, BUFFER_THRESH))
        print('video_size:', self.video_size[-1], 'arrive_time:', self.arrival_time[-1], 'service_time:',
              self.service_time[-1], 'buffer_levels', buffer_levels[-1])
        return buffer_levels

    def calculate_stall_probability(self, buffer_levels):
        stalls = sum(level <= 0 for level in buffer_levels)
        return stalls / len(buffer_levels)

    def set_video_times(self, total_video_time):
        self.total_video_time = total_video_time

    def time_to_frames(self, time_seconds):
        # 将时间（秒）转换为帧数
        frames = int(time_seconds * self.fps)
        return frames
    

    def get_next_features(self):
        # Get the features for the next video chunk period
        start_time = self.video_time / 1000
        # Get the features for the next 10s, if exceed then get till end of video
        end_time = (self.video_time + 10000) / 1000 if self.video_time + 10000 < self.total_video_time \
                                                    else self.total_video_time / 1000
        # crop the video to the specific time range
        video_out_path = "temp_video_for_features.mp4"
        cropped_video = self.video.crop_video(start_time, end_time, video_out_path)

        # Calculate the video features for the cropped video
        # dict of video features [Average SI, Average TI, Average GLCM]
        video_features = self.process_video(video_out_path)


        # Get the trace histogram as the network condition features
        histogram, edge = np.histogram(self.cooked_bw, bins=S_LEN, range=(0, np.max(self.cooked_bw)), density=True)

        # Create a new figure and plot the histogram
        fig, ax = plt.subplots()
        ax.plot(histogram)
        ax.set_title('Network Condition Trace Histogram')
        ax.set_xlabel('Bandwidth')
        ax.set_ylabel('Bin Count')

        # Convert the plot to a PIL Image
        fig.canvas.draw()
        image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

        # Resize the image using a resampling filter
        image = image.resize((640, 480), resample=Image.LANCZOS)  # or Image.Resampling.BILINEAR

        # Convert the PIL Image to a NumPy array
        image_array = np.array(image)

        # Add the image to TensorBoard
        self.writer.add_image('Episode Trace Histogram', image_array, self.eposide, dataformats='HWC')

        # Close the figure to free up memory
        plt.close(fig)

        # normalize the histogram edge
        edge = edge / np.max(self.cooked_bw)

        return video_features, histogram, edge[1:]





    def calculate_si(self,frame):
        """Calculate the Spatial Information (SI) of a frame."""
        laplacian = cv2.Laplacian(frame, cv2.CV_64F)
        si = laplacian.var()
        return si

    def calculate_ti(self,prev_frame, current_frame):
        """Calculate the Temporal Information (TI) between two frames."""
        diff = cv2.absdiff(current_frame, prev_frame)
        ti = diff.var()
        return ti

    def calculate_glcm_features(self,frame):
        """Calculate GLCM properties for a frame."""
        glcm = graycomatrix(frame, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        features = {
            'contrast': graycoprops(glcm, 'contrast')[0, 0],
            'dissimilarity': graycoprops(glcm, 'dissimilarity')[0, 0],
            'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
            'energy': graycoprops(glcm, 'energy')[0, 0],
            'correlation': graycoprops(glcm, 'correlation')[0, 0],
            'ASM': graycoprops(glcm, 'ASM')[0, 0]
        }
        return features

    def extract_frames(self,video_path, num_frames=100):
        """Extract frames from the video for feature calculation."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = total_frames // num_frames
        
        for i in range(num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
            else:
                break
        cap.release()
        return frames

    def process_frame_features(self,frame):
        """Process a single frame to extract SI and GLCM features."""
        si = self.calculate_si(frame)
        glcm_features = self.calculate_glcm_features(frame)
        return si, glcm_features

    def process_video(self,video_path):
        """Process a video to extract and average SI, TI, and GLCM features."""
        frames = self.extract_frames(video_path)
        with ThreadPoolExecutor(max_workers=8) as executor:
            frame_features = list(executor.map(self.process_frame_features, frames))

        si_values, glcm_dicts = zip(*frame_features)
        average_si = np.mean(si_values)
        average_glcm = {key: np.mean([d[key] for d in glcm_dicts]) for key in glcm_dicts[0].keys()}

        ti_values = [self.calculate_ti(frames[i], frames[i+1]) for i in range(len(frames)-1)]
        average_ti = np.mean(ti_values)

        return {
            'Average SI': average_si,
            'Average TI': average_ti,
            'Average GLCM': average_glcm
        }


