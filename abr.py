import sys,logging
import math
CUSHION_SEGMENTS = 5

class Abr():
    def __init__(self):
        #self.logger = logging.getLogger("ABRController.ABR")
        #self.hardcoded_bitrates = (5500, 8500) 
        self.hardcoded_bitrates = (400, 8000) 

    def copy(self):
        # stateless
        return self

    def debug_print(self):
        return "A{} => {}".format(id(self), self.__dict__)

    
    def abr(self, simstate,bitrate_ladder):
        # SEGMENT_DURATION = 4  # seconds
        # RESEVOIR_TEMP = min(5, max(2, 2 * RESEVOIR_EXPANSION_FACTOR_RATE))
        # CUSHION_TEMP = 20  # seconds
        # BITRATE_LADDER = [200, 400, 800, 1200, 2000, 4000, 8000]  # kbps
        # BITRATE_EXPANSION_FACTOR_RESEVOIR_LB = BITRATE_LADDER[0] / 1000 / AVERAGE_BR_LB
        # BITRATE_EXPANSION_FACTOR_RESEVOIR_UB = BITRATE_LADDER[-1] / 1000 / AVERAGE_BR_UB
        # RESEVOIR_EXPANSION_FACTOR_RATE = BITRATE_EXPANSION_FACTOR_RESEVOIR_LB

        # The lower and upper bounds of the average bitrate in kb.
        # 400kbps to 8000kbps (8Mbps)
        AVERAGE_BR_LB = 400
        AVERAGE_BR_UB = 8000
        # The expansion factors for the lower and upper bitrate bounds.
        BITRATE_EXPANSION_FACTOR_RESEVOIR_LB = bitrate_ladder[0]/1000 / AVERAGE_BR_LB
        BITRATE_EXPANSION_FACTOR_RESEVOIR_UB = bitrate_ladder[-1]/1000 / AVERAGE_BR_UB

        # The expansion factor for the reservoir.
        RESEVOIR_EXPANSION_FACTOR_RATE = BITRATE_EXPANSION_FACTOR_RESEVOIR_LB
            
        # used to determine the appropriate bitrate selection
        RESEVOIR_TEMP = min(5, max(2, 2 * RESEVOIR_EXPANSION_FACTOR_RATE))
        CUSHION_TEMP = CUSHION_SEGMENTS * 4.0

        buffer_size = simstate.buffer_size/1000 # in seconds
        #last_level = simstate.history[-1]['level']

        #resolutions = [level['resolution'] for level in VIDEO_PROPERTIES[chunk_index]['levels']]
        # number of resolutions in the chunk
        n_resolutions = len(bitrate_ladder)

        if buffer_size < RESEVOIR_TEMP:
            quality_level = 0
        elif buffer_size >= RESEVOIR_TEMP + CUSHION_TEMP:
            quality_level = n_resolutions - 1
        else:
            #Dividing this available buffer space by the CUSHION_TEMP gives a value between 0 and 1, 
            #representing the position of the buffer size within the cushion.
            f_buf_now = (n_resolutions - 1) * (buffer_size - RESEVOIR_TEMP) / float(CUSHION_TEMP)
            resolution_index = int(f_buf_now)
            quality_level = resolution_index

        return quality_level