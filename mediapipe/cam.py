import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import ktb

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

k = ktb.Kinect()

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        frame = k.get_frame(ktb.RAW_COLOR)
        
        frame = cv2.resize(frame, dsize = (0, 0), fx = 0.5, fy = 0.5)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.flip(frame, 1)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #mask = np.zeros(frame.shape, np.uint8)
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )
                                 
        cv2.imshow('mediapipe pose', image)
        if (cv2.waitKey(10) & 0xFF == ord('q')) or (cv2.waitKey(10) & 0xFF == 27):
            break

    cv2.destroyAllWindows()
