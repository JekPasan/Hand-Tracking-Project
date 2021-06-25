import cv2
import time
import numpy as np
import hand_tracking_module as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap = cv2.VideoCapture(0)

current_time = 0
previosu_time = 0
fps = 0

detector = htm.HandDetector(max_num_hands=1, detection_confidance=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume_range = volume.GetVolumeRange()
max_volume = volume_range[1]
min_volume = volume_range[0]
hand_volume = 0

while True:
    success, img = cap.read()

    img = detector.find_hands(img, draw=False)
    landmarks = detector.find_position(img, draw=False)

    if len(landmarks) != 0:
        thumbX, thumbY = landmarks[4][1], landmarks[4][2]
        indexX, indexY = landmarks[8][1], landmarks[8][2]
        line_center = [(thumbX+indexX)//2, (thumbY + indexY)//2]
        cv2.circle(img, (thumbX, thumbY), 8, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (indexX, indexY), 8, (0, 0, 255), cv2.FILLED)
        cv2.line(img, (thumbX, thumbY), (indexX, indexY), (0, 0, 255), 3)

        length = math.hypot(indexX-thumbX, indexY-thumbY)
        hand_volume = np.interp(length, [40, 367], [min_volume, max_volume])
        volume.SetMasterVolumeLevel(hand_volume, None)

        if length <= 40:
            cv2.circle(img, (line_center[0], line_center[1]), 8, (255, 0, 90), cv2.FILLED)
        else:
            cv2.circle(img, (line_center[0], line_center[1]), 8, (0, 0, 255), cv2.FILLED)

    current_time = time.time()
    fps = int(1/(current_time-previosu_time))
    previosu_time = current_time
    cv2.putText(img, str(fps), (5, 35), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)

    cv2.imshow("Video", img)
    cv2.waitKey(1)