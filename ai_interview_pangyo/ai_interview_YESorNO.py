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

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

def putText(image, x, coord):
    cv2.putText(image, x, coord, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(image, x, coord, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)

frame_num = 0
frame_sec = 30
frame_list = []
result_action = 0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True: #cap.isOpened()
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
            
            l_index = pose_landmarks(landmarks, 'LEFT_INDEX')
            r_index = pose_landmarks(landmarks, 'RIGHT_INDEX')
            
            #사용자 기준 좌측을 YES, 우측을 NO
            this_action = 'NONE'
            
            if l_index[1] < 0.4 : #오른손 사용
                if l_index[0] < 0.5 : #사용자 기준 좌측
                    this_action = 'YES'
                    frame_num += 1
                    frame_list.append(-1)
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																									
                    if frame_list[-2] == 1 :
                        frame_num = 0
                    else :
                        if frame_num == frame_sec :
                            result_action = -1 #TCP
                            frame_num = 0
            
                else : #사용자 기준 우측
                    this_action = 'NO'
                    frame_num += 1
                    frame_list.append(1)
                    if frame_list[-2] == -1 :
                        frame_num = 0
                    else :
                        if frame_num == frame_sec :
                            result_action = 1 #TCP
                            frame_num = 0
                    
                    
            elif r_index[1] < 0.4 : #왼손 사용
                if r_index[0] < 0.5 : #사용자 기준 좌측
                    this_action = 'YES'
                    frame_num += 1
                    frame_list.append(-1)
                    if frame_list[-2] == 1 :
                        frame_num = 0
                    else :
                        if frame_num == frame_sec :
                            result_action = -1 #TCP
                            frame_num = 0
            
                else : #사용자 기준 우측
                    this_action = 'NO'
                    frame_num += 1
                    frame_list.append(1)
                    if frame_list[-2] == -1 :
                        frame_num = 0
                    else :
                        if frame_num == frame_sec :
                            result_action = 1 #TCP
                            frame_num = 0
            
            else :
                frame_num = 0
                
            #TCP : result_action, frame_num(?)
                    
                    
            putText(image, "state: {} ".format(str(this_action)), (20,50))
            putText(image, "frame_num: {} ".format(str(frame_num)), (20,100))
            putText(image, "result: {} ".format(str(result_action)), (20,150))
            #putText(image, "l_index: {} ".format(str(l_index)), (20,100))
            #putText(image, "r_index: {} ".format(str(r_index)), (20,150))
            
            
        except:
            pass
            
        cv2.imshow('Pose Estimation', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    #cap.release()
    cv2.destroyAllWindows()

