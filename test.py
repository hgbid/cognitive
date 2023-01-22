"""""
from time import sleep
from tqdm import tqdm
for i in tqdm(range(10)):
    sleep(3)
"""
from matplotlib import pyplot as plt

"""""
from moviepy.editor import *

clip1 = VideoFileClip("geeks.mp4").subclip(0, 5).margin(10)

clip2 = clip1.fx(vfx.mirror_x)

clip3 = clip1.fx(vfx.mirror_y)

clip4 = clip1.resize(0.60)

# clips list
clips = [[clip1, clip2],
         [clip3, clip4]]

# stacking clips
final = clips_array(clips)

# showing final clip
final.ipython_display(width=480)
"""


import cv2
cap = cv2.VideoCapture("\file.mp4")
cap.set(cv2.cv.CV_CAP_PROP_FPS, 15)
