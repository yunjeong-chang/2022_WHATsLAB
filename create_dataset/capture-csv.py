import mediapipe as mp
import cv2
import numpy as np
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
    
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].x)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].z)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility)
    
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility)
    
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL.value].x)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL.value].y)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL.value].z)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL.value].visibility)
    
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL.value].x)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL.value].y)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL.value].z)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL.value].visibility)
    
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].visibility)
    
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].visibility)
    
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].x)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].z)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility)
    
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility)
    
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].x)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].z)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility)
    
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].x)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].z)
    pose.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility)
    
    return pose


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

k = ktb.Kinect()

landmarks = ['class']
num_coords = 16

for val in range(1, num_coords+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

#run only the first time and comment out
with open('dataset-metaverse.csv',mode='w',newline='') as f :
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)
    
class_name = "run"
flag = 0 

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

        if flag < 100 :
            cv2.putText(image, 'STARTING COLLECTION {}'.format(flag), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_4)
            cv2.putText(image, '{}'.format(class_name), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2, cv2.LINE_4)
            cv2.imshow('mediapipe pose', image)
        elif flag < 500 :                
            row = get_pose_conf_pick(results)
            row.insert(0, class_name)
            with open('dataset-metaverse.csv', mode='a',newline='') as f :
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)
            cv2.putText(image, '{}'.format(flag), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_4)
            cv2.imshow('mediapipe pose', image)
        else :
            break
        
        flag += 1           

        if (cv2.waitKey(10) & 0xFF == ord('q')) or (cv2.waitKey(10) & 0xFF == 27):
            break

    cv2.destroyAllWindows()
