# -----------------------------------------
# ---- the bare minimum necessary code ----
# -----------------------------------------

import cv2
import mediapipe as mp
import time

# inputing the video capture device
cap = cv2.VideoCapture(0)

# importing the modules needed to identify the hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# importing the function for drawing the hands' joints
mp_draw = mp.solutions.drawing_utils

# some variables used for displaying the fps
previous_time = 0
current_time = 0

while True:
    # getting all the data needed from the capture device
    success, img = cap.read()

    # the rgb values of the displaying image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # getting all the hands' joints from the image
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmark.landmark):
                # getting the height, width and channel of the screen
                h, w, c = img.shape
                # getting the X and Y position of the joints
                cX, cY = int(lm.x * w), int(lm.y * h)
                
                # drawing circles on the tip of the fingers
                if id == 4 or id == 8 or id == 12 or id == 16 or id == 20:
                    cv2.circle(img, (cX,cY), 6, (132, 44, 76), cv2.FILLED)
            
            # drawing all the hands' joints and connecting them
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)
    
    # getting and displaying the fps
    current_time = time.time()
    fps = int(1/(current_time-previous_time))
    previous_time = current_time
    cv2.putText(img, str(fps), (5, 35), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)

    # displaying the webcam to the screen
    cv2.imshow("Video", img)
    cv2.waitKey(1)