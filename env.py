import numpy as np
import core as abrenv
import load_trace
from video import Video
from core import S_INFO, S_LEN,A_DIM 
# download time in seconds,buffer size in 10 sec, throughput in kbps, chunk duration in ms, chunk CRF
# video features: SI, TI, GLCM, network condition histogram
#S_INFO = 9
#S_LEN = 8  # take how many frames in the past
#A_DIM = 2
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100

BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
EPS = 1e-6


class ABREnv():
    def __init__(self, random_seed=RANDOM_SEED, trace_folder=None,video_path=None,writer=None, test=False):
        if test:
            trace_folder = './test/' if trace_folder is None else trace_folder
        # trace_folder = './train/' if trace_folder is None else trace_folder
        video_path = 'bigbuckbunny360p24.mp4' if video_path is None else video_path
        np.random.seed(random_seed)
        all_cooked_time, all_cooked_bw, all_trace_file_names = load_trace.load_trace()
        self.net_env = abrenv.Environment(all_cooked_time=all_cooked_time,
                                          all_cooked_bw=all_cooked_bw,
                                          random_seed=random_seed,
                                          video=Video(video_path, logdir='log'),
                                          #video=Video('bigbuckbunny2160p60.mp4', logdir='log'),
                                          writer = writer
                                          )

        self.last_bit_rate = DEFAULT_QUALITY
        self.buffer_size = 0.
        self.alpha = 0.02
        self.beta = -0.10
        self.gamma = -1.0
        self.delta = 0.01
        self.epsilon = 0.01
        self.B = []
        self.CRF = []
        self.state = np.zeros((S_INFO, S_LEN))

    def seed(self, num):
        np.random.seed(num)

    def reset(self):
        # self.net_env.reset_ptr()
        self.time = 0
        self.time_stamp = 0
        self.state = np.zeros((S_INFO, S_LEN))
        self.buffer_size = 0.
        state = np.roll(self.state, -1, axis=1) # shift left ex. [1,2,3,4,5] -> [2,3,4,5,0]
        self.state = state
        return state

    def render(self):
        pass

    def close(self):
        pass

    def step(self, action):
        self.time += 1
        B, CRF = decoder_action(action)
        self.B.append(B)
        self.CRF.append(CRF)
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, self.buffer_size, rebuf, \
        video_chunk_size, end_of_video, pst, vmaf, last_vmaf, reward = self.net_env.get_video_chunk(B, CRF)
        print('duration:', B,'ms ', 'CRF:', CRF, 'video_time_now:', self.net_env.video_time / 1000, 'vmaf:', vmaf, 'pst:', pst)
        # reward is video quality - smooth penalty - rebuffer penalty

        
        # get the new video feature state after taking the action
        video_features, network_condition_histogram, edge = self.net_env.get_next_features()

        # Extract the individual video features
        avg_si = video_features['Average SI']
        avg_ti = video_features['Average TI']
        avg_glcm = video_features['Average GLCM']
        avg_contrast = avg_glcm['contrast']
        avg_dissimilarity = avg_glcm['dissimilarity']
        avg_homogeneity = avg_glcm['homogeneity']
        avg_energy = avg_glcm['energy']
        avg_correlation = avg_glcm['correlation']
        avg_ASM = avg_glcm['ASM']

        # Shift the state array to the left to make room for the new features
        self.state = np.roll(self.state, -1, axis=1)

        # Update the state with the new features
        # State shape: (S_INFO, S_LEN) -> (10,10)
        self.state[0, -1] = delay / 1000.
        self.state[1, -1] = self.buffer_size / BUFFER_NORM_FACTOR
        self.state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K
        self.state[3, -1] = self.B[self.time - 1] / 1000.
        self.state[4, -1] = self.CRF[self.time - 1]
        self.state[5, -1] = avg_si
        self.state[6, -1] = avg_ti
        #print shape
        self.state[7, -6:] = [avg_contrast, avg_dissimilarity, avg_homogeneity, avg_energy, avg_correlation, avg_ASM]
        #print('shape glacm',self.state[7, -6:].shape)
        #print('network_condition_histogram shape:', network_condition_histogram.shape)
        self.state[8:, :] = network_condition_histogram
        #print('edge shape:', edge.shape)
        #print('edge shape:', edge.reshape(-1, S_LEN).shape)
        self.state[9, :] = edge

        # Flatten the state array before returning it
        set_state = self.state
        # observation, reward, done, info = env.step(action)
        info = None
        return set_state, reward, end_of_video, info


def linear_mapping(value, in_min, in_max, out_min, out_max):
    # 线性映射公式
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def decoder_action(action):
    B, CRF = (action[0] + 1), (action[1] + 1)
    B = linear_mapping(B, 0, 2, 500, 6000)
    CRF = round(linear_mapping(CRF, 0, 2, 15, 40)) # 0-51 too large -> 15-40 try
    return B, CRF


print(decoder_action([1, 1]))
