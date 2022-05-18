import cv2
import os
import numpy as np
import mediapipe as mp 
import ktb
import timeit
import math
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

k = ktb.Kinect()
#cap = cv2.VideoCapture(0)

def pose_landmarks(landmarks, location):
    x = landmarks[mp_pose.PoseLandmark[location].value].x
    y = landmarks[mp_pose.PoseLandmark[location].value].y
    z = landmarks[mp_pose.PoseLandmark[location].value].z
    
    x = round(x,2)
    y = round(y,2)
    z = round(z,2)
    
    return [x,y,z]

def putText(image, x, coord):
    cv2.putText(image, x, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(image, x, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
action_list = []
    
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True: 
        frame = k.get_frame(ktb.RAW_COLOR)
        #ret, frame = cap.read()
        
        frame = cv2.resize(frame, dsize = (0, 0), fx = 0.5, fy = 0.5) 
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.flip(frame, 1)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        image.flags.writeable = False                  
        results = pose.process(image)                  
        image.flags.writeable = True                   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        mask = np.zeros(frame.shape, np.uint8)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                  )
        try :

            landmarks = results.pose_landmarks.landmark

            l_knee = pose_landmarks(landmarks, 'LEFT_KNEE')
            r_knee = pose_landmarks(landmarks, 'RIGHT_KNEE')
            l_ankle = pose_landmarks(landmarks, 'LEFT_ANKLE')
            r_ankle = pose_landmarks(landmarks, 'RIGHT_ANKLE')
            l_heel = pose_landmarks(landmarks, 'LEFT_HEEL')
            r_heel = pose_landmarks(landmarks, 'RIGHT_HEEL')
            l_foot_index = pose_landmarks(landmarks, 'LEFT_FOOT_INDEX')
            r_foot_index = pose_landmarks(landmarks, 'RIGHT_FOOT_INDEX')
            y = round(abs((l_knee[1]+l_ankle[1]+l_heel[1])-(r_knee[1]+r_ankle[1]+r_heel[1])),3)
            
            if y < 0.04 : 
                action_list.append(0) #stop  
            else :
                action_list.append(1) #walk

            action = sum(action_list[-50:]) / 50
            
            this_action = ""
            
            if action < 0.01 : 
                this_action = "stop"
            else :
                this_action = "walk"
                      
            putText(image, "state: {} ".format(str(this_action)), (20,50))  
            putText(image, "action: {} ".format(str(action)), (20,100)) 
            putText(image, "y: {} ".format(y), (20,150))
            
            #putText(image, "knee_y: {} ".format(round(l_knee[1]-r_knee[1],3)), (20,300))  
            #putText(image, "ankle_y: {} ".format(round(l_ankle[1]-r_ankle[1],3)), (20,350)) 
            #putText(image, "heel_y: {} ".format(round(l_heel[1]-r_heel[1],3)), (20,400)) 

        except:
            pass
              
        cv2.imshow('Metaverse', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    #cap.release()
    cv2.destroyAllWindows()

