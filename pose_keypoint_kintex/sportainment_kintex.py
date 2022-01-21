import cv2
import os
import time
import numpy as np
import mediapipe as mp 
from matplotlib import pyplot as plt
import ktb
import copy

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

actions = np.array(['center', 'backward', 'left', 'right', 'forward'])       

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

#cap = cv2.VideoCapture(1)
k = ktb.Kinect()

this_action = '?'
action_seq1 = []
action_seq2 = []

# Set mediapipe model 
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        #ret, frame = cap.read()
        frame = k.get_frame(ktb.RAW_COLOR)
        #frame = cv2.resize(frame, (960, 820)) #1280
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
        try :

            landmarks = results.pose_landmarks.landmark

            nose=[landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y,landmarks[mp_pose.PoseLandmark.NOSE.value].z]
            l_shoulder=[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
            l_hip=[landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
            l_knee=[landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
            l_ankle=[landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
            l_elbow=[landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
            l_foot=[landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y,
             landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z]
            r_shoulder=[landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
            r_hip=[landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
            r_knee=[landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
            r_ankle=[landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
            r_elbow=[landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
            r_foot=[landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y,
                landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z]

#             l_shoulder_angle = calculate_angle(l_elbow, l_shoulder, l_hip)
#             l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
#             l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
#             l_ankle_angle = calculate_angle(l_knee, l_ankle, l_foot)
            
#             r_shoulder_angle = calculate_angle(r_elbow, r_shoulder, r_hip)
#             r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)
#             r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
#             r_ankle_angle = calculate_angle(r_knee, r_ankle, r_foot)   
            
            this_action = '?'
            action_seq1_mean = 0
            action_seq2_mean = 0
            
            action_seq1.append(nose[2]-l_foot[2])
            action_seq2.append(l_hip[0]-r_shoulder[0])
            
            action_seq1_mean = np.mean(np.array(action_seq1[-5:]))
            action_seq2_mean = np.mean(np.array(action_seq2[-5:]))
            
            tmp_list = []
            y = (action_seq1_mean + 0.1) * -2.5
            x1 = (action_seq2_mean - 0.15) * 20
            x2 = (action_seq2_mean - 0.225) * 20
            if y > 1 : y = 1
            if y < -1: y = -1
            if x1 > 1 : x1 = 1
            if x1 < -1: x1 = -1
            if x2 > 1 : x2 = 1
            if x2 < -1: x2 = -1
            tmp_list.append(y)
            
            if y > 0.5:
                tmp_list.append(x2)
            else:
                tmp_list.append(x1)
            print(tmp_list)

                           
            if (action_seq1_mean) < -0.5 :
                if (action_seq2_mean) < 0.15 :
                    this_action = 'Left_Forward'
                    
                elif (action_seq2_mean) > 0.3 :
                    this_action = 'Right_Forward'
                else :
                    this_action = 'Forward'
                    
            elif (action_seq1_mean) > 0.3 :
                if (action_seq2_mean) < 0.1 :
                    this_action = 'Left_Backward'
                elif (action_seq2_mean) > 0.2 :
                    this_action = 'Right_Backward'
                else :
                    this_action = 'Backward'
            
            else :
                if (action_seq2_mean) < 0.1 :
                    this_action = 'Left'
                elif (action_seq2_mean) > 0.2 :
                    this_action = 'Right'
                else :
                    this_action = 'Center'

            
            cv2.putText(image, "state1: {}".format(str(this_action)), (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, "y: {}".format((y)), (20,150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, "x1: {}".format((x1)), (20,200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, "x2: {}".format((x2)), (20,250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2, cv2.LINE_AA)
                    
        
        except:
            pass
        
        cv2.imshow('Pose Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
                    
    #cap.release()
    cv2.destroyAllWindows()
