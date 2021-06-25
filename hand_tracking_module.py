# --------------------
# ---- the module ----
# --------------------

import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, static_mode=False, max_num_hands=2, detection_confidance=0.5, track_confidence=0.5):
        self.static_mode = static_mode
        self.max_num_hands = max_num_hands
        self.detection_confidence = detection_confidance
        self.track_confidence = track_confidence
        
        self.cap = cv2.VideoCapture(0)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.static_mode, self.max_num_hands, self.detection_confidence, self.track_confidence)

        self.mp_draw = mp.solutions.drawing_utils
    
    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks:
            for hand_landmark in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmark, self.mp_hands.HAND_CONNECTIONS)
        return img
    
    def find_position(self, img, hand_number=0, draw=True):
        # position of all the landmarks
        lm_list = []

        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_number]
            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cX, cY = int(lm.x * w), int(lm.y * h)

                lm_list.append([id, cX, cY])
                if draw and (id == 4 or id == 8 or id == 12 or id == 16 or id == 20):
                    cv2.circle(img, (cX,cY), 6, (132, 44, 76), cv2.FILLED)
        
        return lm_list



def main():
    # some variables used for displaying the fps
    previous_time = 0
    current_time = 0
    fps = 0

    # inputing the video capture device
    cap = cv2.VideoCapture(0)

    detector = HandDetector()

    while True:
        # getting all the data needed from the capture device
        success, img = cap.read()
        img = detector.find_hands(img)
        lm_list = detector.find_position(img)

        if len(lm_list) != 0:
            print(lm_list[7])

        # getting and displaying the fps
        current_time = time.time()
        fps = int(1/(current_time-previous_time))
        previous_time = current_time
        cv2.putText(img, str(fps), (5, 35), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)

        # displaying the webcam to the screen
        cv2.imshow("Video", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()