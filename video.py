import json
import os
import subprocess
import sys
from logging_segue import create_logger
from moviepy.editor import VideoFileClip


B_IN_BYTE = 8
M_IN_K = 1000
K_RESCALING_METHOD_KEYFRAMES_APPROACH = "keyframes_method"

K_RESCALING_METHOD_KEYFRAMES_GOP = "gop"
K_RESCALING_METHOD_KEYFRAMES_CONSTANT = "constant"
K_RESCALING_METHOD_KEYFRAMES_FORCE_KEYS = "force_keys"

K_RESCALING_METHOD_GOP_SECONDS = "gop_seconds"
K_RESCALING_METHOD_SEGMENT_SECONDS = "segment_seconds"
K_RESCALING_METHOD_FORCED_INDEXES_LIST = "keyframes_indexes_list"
K_RESCALING_METHOD_FORCED_TIMESTAMPS_LIST = "keyframe_timestamps_list"


def pmkdir(kdir):
    if not os.path.exists(kdir):
        os.makedirs(kdir)


class Video:
    """
    A wrapper around ffmpeg to retrieve and manipulate video information.
    
    This class provides functionalities to:
    - Retrieve video details (like fps, duration, resolution etc.)
    - Rescale videos at different resolutions.
    - Segment videos based on keyframes.
    """

    ''' Attributes: fps, duration, total_frames, bytes, resolution, 
        bitrate, keyframes_index_list, keyframes_timestamp_list'''

    '''Utility function include: load_fps, load_duration, load_total_frames, load_bytes, 
                                 load_resolution, load_bitrate, load_keyframes_indexes, 
                                 load_keyframes_timestamps, load_keyframes, dump_key_cache, read_key_cache
                                 rescale_h264_constant_quality, rescale_at_resolution,
                                 check_other_video, video_path, get_video_stats,
                                 rescale_at_res_method_switching,  rescale_h264_two_pass, rescale_vp9_two_pass,
                                '''

    def __init__(self, video_path, logdir, verbose=False, concat_file=None):
        """
        Initializes the Video object.
        
        :param video_path: Path to the video file.
        :param logdir: Directory for storing logs.
        :param verbose: Flag indicating verbosity level.
        :param concat_file: File used for concatenating videos (if needed).
        """

        self.logs_dir = logdir
        self.verbose = verbose
        pmkdir(logdir)
        video_id = video_path.replace('/', '-')
        log_file = os.path.join(logdir, 'video_{}.log'.format(video_id))
        self.logger = create_logger('Video {}'.format(video_id), log_file, verbose=verbose)

        if not os.path.exists(video_path):
            self.logger.info("Video {} does not exist.".format(video_path))
            if concat_file:
                self.logger.info("Concatenation specified: creating video")
                concat_cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_file, '-c', 'copy', '-y', video_path]
                concat_cmd_formatted = ' '.join(concat_cmd).strip()
                self.logger.debug("Executing {}".format(concat_cmd_formatted))
                proc = subprocess.Popen(concat_cmd_formatted, shell=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT)
                outs, errs = proc.communicate()
                assert os.path.exists(video_path)

            else:
                self.logger.error("Video does not exist and concatenation not specified")
                sys.exit(-1)

        else:
            self.logger.debug("Video {} exists".format(video_path))
            self.logger.debug("Video object has been created correctly")

        self._video_path = video_path

        ## lazy: information are loaded only if the proper function is called

        ## general info
        self._fps = None
        self._duration = None
        self._total_frames = None
        self._bytes = None
        self._resolution = None
        self._bitrate = None
        self._keyframes_index_list = None
        self._keyframes_timestamp_list = None

    def video_path(self):
        """
        Returns the video path associated with the Video object.
        """
        return self._video_path
    def video_name(self):
        """
        Returns the video name associated with the Video object.
        """
        #ex: /home/ly/ABR/ABR-PPO-Pytorch/src-2/bigbuckbunny360p24.mp4 -> bigbuckbunny360p24.mp4
        return self._video_path.split('/')[-1]

    def load_resolution(self):
        """
        Loads and returns the resolution of the video.
        """
        if self._resolution:
            self.logger.debug("Already computed resolution: {}".format(self._resolution))
            return self._resolution

        result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                 "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0",
                                 self._video_path],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)

        self._resolution = result.stdout.decode("utf-8").rstrip()
        return self._resolution

    def load_total_frames(self):
        """
        Calculates and returns the total number of frames in the video.
        """
        if self._total_frames:
            self.logger.debug("Already computed total frames: {}".format(self._total_frames))
            return self._total_frames

        self.logger.debug("For frames count, fetching container")
        result = subprocess.run(["ffprobe", "-v", "error",
                                 "-select_streams", "v:0", "-show_entries",
                                 "stream=nb_frames", "-of", "default=nokey=1:noprint_wrappers=1", self._video_path],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        try:
            result = float(result.stdout)
        except:
            self.logger.debug("Fetching container failed. Computing invoking ffprobe")
            result = subprocess.run(["ffprobe", "-v", "error",
                                     "-count_frames", "-select_streams", "v:0", "-show_entries",
                                     "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1",
                                     self._video_path],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
            result = float(result.stdout)

        self.logger.debug("Video {}: Computed a total of {} frames".format(self._video_path, result))

        self._total_frames = result
        return result

    def load_fps(self):
        """
        Loads and returns the frames-per-second (FPS) of the video.
        """

        if self._fps:
            self.logger.debug("Already computed FPS: {}".format(self._fps))
            return self._fps

        result = subprocess.run(["ffprobe", "-v", "error",
                                 "-select_streams", "v", "-of",
                                 "default=noprint_wrappers=1:nokey=1",
                                 "-show_entries", "stream=r_frame_rate", self._video_path],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)

        result = result.stdout.decode('utf-8')
        result = float(result.split('/')[0]) / float(result.split('/')[1])
        result = "{:.2f}".format(result)
        self._fps = float(result)
        self.logger.debug("Video {} is {} fps".format(self._video_path, self._fps))
        return self._fps

    def load_duration(self):
        """
        Loads and returns the duration (in seconds) of the video.
        """
        if self._duration:
            self.logger.debug("Already computed duration: {}".format(self._duration))
            return self._duration

        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                 "format=duration", "-of",
                                 "default=noprint_wrappers=1:nokey=1", self._video_path],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)

        self._duration = float(result.stdout)
        return self._duration

    def load_bytes(self):
        """
        Loads and returns the size (in bytes) of the video.
        """
        if self._bytes:
            self.logger.debug("Already computed bytes: {}".format(self._bytes))
            return self._bytes

        self._bytes = os.path.getsize(self._video_path)
        return self._bytes

    def load_bitrate(self):
        """
        Calculates and returns the bitrate of the video.
        """
        if self._bitrate:
            self.logger.debug("Already computed bitrate: {}".format(self._bitrate))
            return self._bitrate

        self._bitrate = self.load_bytes() * B_IN_BYTE / self.load_duration() / M_IN_K
        return self._bitrate

    def get_video_stats(self):
        """
        Retrieves and returns a tuple of video stats: duration, bitrate, and size in bytes.
        """
        return self.load_duration(), self.load_bitrate(), self.load_bytes()

    def check_other_video(self, other_video_path, force):
        """
        Checks the existence and validity of another video. If force is True, overwrites the existing video.

        :param other_video_path: Path to the other video.
        :param force: Flag indicating if overwriting is allowed.
        """
        if os.path.exists(other_video_path) and not force:
            self.logger.debug("Video exists and force option deselected")
            other_video = Video(other_video_path, self.logs_dir, self.verbose)
            try:
                if (other_video.load_total_frames() != self.load_total_frames()):
                    self.logger.info("Video {} exists but corrupted. ".format(other_video_path))
                    self.logger.info(
                        "Total frames: {}, Expected {}. Recomputing...".format(other_video.load_total_frames(),
                                                                               self.load_total_frames()))
                    os.remove(other_video_path)
                    return None
            except:
                self.logger.info("Video is not well formed. Removing")
                os.remove(other_video_path)
                return None

            else:
                self.logger.info("Video is well formed. Skipping computation")
                return other_video

        if force and os.path.exists(other_video_path):
            self.logger.debug("Video {} exists and force option selected. Removing video..".format(other_video_path))
            os.remove(other_video_path)
            return None

        return None

    def rescale_h264_constant_quality(self,
                                      video_out_path,
                                      crf,
                                      width,
                                      height,
                                      gop=0,
                                      forced_key_frames=None,
                                      force=False):
        """
        Rescales the video using h264 codec with constant quality.
        
        :param video_out_path: Output path for the rescaled video.
        :param crf: Constant Rate Factor for the video encoding.
        :param width: Target width of the video.
        :param height: Target height of the video.
        :param gop: Group of pictures size.
        :param start_frame: Start frame for video encoding (optional).
        :param end_frame: End frame for video encoding (optional).
        :param forced_key_frames: List of frames to be forced as keyframes.
        :param force: Flag to force the re-encoding even if output exists.
        """
        '''Rescale a video at a given resolution, using ffmpeg and h264 codec'''
        '''If force_key_frames is not None, the video is rescaled using the keyframes 
        specified which will serve as boundaries for the segments'''

        self.logger.debug("Selected rescale method h264 in constant quality")

        video = self.check_other_video(video_out_path, force)
        if video is not None:
            return video

        rescale_string = "-s {}x{}".format(width, height)

        gop_string = ""
        if gop > 0:
            self.logger.debug("Rescaling with a gop of {}".format(gop))
            gop_string = "-g {}".format(gop)

        forced_key_frames_string = ""

        if forced_key_frames is not None:
            if isinstance(forced_key_frames, list):
                self.logger.debug("List of keyframes specified")
                k_string = ['eq(n,{})'.format(k) for k in forced_key_frames]
                k_string = '+'.join(k_string).strip()
                forced_key_frames_string = '-force_key_frames "expr:{}"'.format(k_string)

        cmd = "ffmpeg -i {} -c:v libx264 -crf {} {} {} {} -y {}".format(
            self._video_path,
            crf,
            rescale_string,
            forced_key_frames_string,
            gop_string,
            video_out_path
        )

        self.logger.debug("Executing {}".format(cmd))
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        outs, errs = proc.communicate()
        assert os.path.exists(video_out_path)
        self.logger.debug("Rescaling from {} to {} completed succesfully!".format(self._video_path, video_out_path))
        video = Video(video_out_path, self.logs_dir, self.verbose)

        return video

    # def crop_video(self, start_time, end_time, video_out_path):
    #     cmd = "ffmpeg -i {} -ss {} -to {} -c copy {}".format(self._video_path, start_time, end_time, video_out_path)
    #     print("Executing {}".format(cmd))
    #     proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     outs, errs = proc.communicate()
    #     print("FFmpeg output:", outs.decode())
    #     print("FFmpeg error:", errs.decode())
        
    #     if not os.path.exists(video_out_path):
    #         raise RuntimeError("Cropping failed. Output video file does not exist.")
        
    #     print("Cropping from {} to {} completed successfully!".format(self._video_path, video_out_path))
    #     video = Video(video_out_path, self.logs_dir, self.verbose)
    #     return video
    def crop_video(self, start_time, end_time, video_out_path):
        # 获取视频的帧率
        vid = VideoFileClip(self._video_path)
            # Add a small tolerance value to handle slight duration mismatches
        tolerance = 0.1
        end_time = min(end_time, vid.duration - tolerance)
        try:
            video_clip = vid.subclip(start_time, end_time)
        except:
            print("start_time:", start_time, "end_time:", end_time)
        video_clip.write_videofile(video_out_path)
        video = Video(video_out_path, self.logs_dir, self.verbose)

        return video
