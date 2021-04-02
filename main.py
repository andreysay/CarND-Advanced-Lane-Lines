"""
Lane Lines Detection pipeline

Usage:
    main.py [--video] INPUT_PATH OUTPUT_PATH 

Options:

-h --help                               show this screen
--video                                 process video file instead of image
"""

import numpy as np
import matplotlib.image as mpimg
import cv2
from docopt import docopt
from IPython.display import HTML, Video
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
from Utility import *
from Thresholding import *
from imageWarper import *
from Line import *

class FindLaneLines:
    """ This class is for parameter tunning.

    Attributes:
        ...
    """
    def __init__(self):
        """ Init Application"""
        self.mtx, self.dist = loadCoefficients()
        self.imgSize = (720, 1280)
        # Define 4 source points
        self.src = np.float32([[250, self.imgSize[0]-25], [575, 460], 
                      [700, 460], [1150, self.imgSize[0]-25]])
        # Define 4 destination points
        self.dst = np.float32([[320, self.imgSize[0]-25], [320, 0], 
                      [960, 0], [960, self.imgSize[0]-25]])
        self.thresholding = thresholdPipeline
        self.undistort = undistortImage
        self.transform = warper
        self.laneLines = LaneLines()

    def runPipeline(self, img):
        out_img = np.copy(img)

        img_undist = self.undistort(img, self.mtx, self.dist)
        #cv2.imwrite('output_images/undistorted.jpg', img_undist)
        img = self.thresholding(img_undist)
        #cv2.imwrite('output_images/thresholded.jpg', img)

        binaryWarped, M, invM = self.transform(img, self.src, self.dst)
        #cv2.imwrite('output_images/binarywarped.jpg', binaryWarped)
        self.laneLines.find_lane_pixels(binaryWarped)
        self.laneLines.fit_polynomial(binaryWarped)
        self.laneLines.measure_curvature_real(binaryWarped)

        out_img = self.laneLines.drawLane(img_undist, binaryWarped, invM)
        #cv2.imwrite('output_images/drawlanes.jpg', out_img)
        out_img = self.laneLines.drawData(out_img)

        return out_img

    def processImage(self, input_path, output_path):
        img = cv2.imread(input_path)
        out_img = self.runPipeline(img)
        cv2.imwrite(output_path, out_img)

    def processVideo(self, input_path, output_path):
        clip = VideoFileClip(input_path)
        out_clip = clip.fl_image(self.runPipeline)
        out_clip.write_videofile(output_path, audio=False)

def main():
    args = docopt(__doc__)
    input = args['INPUT_PATH']
    output = args['OUTPUT_PATH']

    clibrateCamera()

    findLaneLines = FindLaneLines()
    if args['--video']:
        findLaneLines.processVideo(input, output)
    else:
        findLaneLines.processImage(input, output)


if __name__ == "__main__":
    main()