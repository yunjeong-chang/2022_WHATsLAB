import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import uuid
import os
import ktb
import csv

def get_pose_conf_pick(results):
    pose = []
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility)
    
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility)
    
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].z)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].visibility)
    
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].z)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility)
    
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].x)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].y)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].z)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility)
    
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].x)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].y)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].z)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility)
    
    return pose


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

k = ktb.Kinect()

import pickle 
with open('model-csv.pkl', 'rb') as f:
    model = pickle.load(f)

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
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )

        try:
            row = get_pose_conf_pick(results)

            # Make Detections
            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            print(body_language_class, body_language_prob)
            
            # Display Probability
            cv2.putText(image, 'PROB', (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2)), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
       
            # Display Class
            cv2.putText(image, 'CLASS', (135,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0], (130,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        except:
            pass
	
        cv2.imshow('mediapipe pose', image)
        if (cv2.waitKey(10) & 0xFF == ord('q')) or (cv2.waitKey(10) & 0xFF == 27):
            break

    cv2.destroyAllWindows()
