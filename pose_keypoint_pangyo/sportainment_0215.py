import cv2
import os
import time
import numpy as np
import mediapipe as mp 
from matplotlib import pyplot as plt
import ktb
import copy

actions = np.array(['center', 'backward', 'left', 'right', 'forward'])       

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

k = ktb.Kinect()

this_action = '?'
action_seq1 = []
action_seq2 = []

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
        try :

            landmarks = results.pose_landmarks.landmark

            nose=[landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y,landmarks[mp_pose.PoseLandmark.NOSE.value].z]
            l_shoulder=[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
            l_hip=[landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
            l_knee=[landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]
            l_foot=[landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z]
            r_shoulder=[landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
            r_hip=[landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
            r_knee=[landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
            r_foot=[landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z] 
            
            if l_knee[2]-r_foot[2] < 0 : #left foot in front
            
                this_action = '?'
                action_seq1_mean = 0
                action_seq2_mean = 0
            
                action_seq1.append(nose[2]-l_foot[2]) #forward or backward
                action_seq2.append(l_hip[0]-r_shoulder[0]) #left or right
            
                action_seq1_mean = np.mean(np.array(action_seq1[-5:])) #forward or backward
                action_seq2_mean = np.mean(np.array(action_seq2[-5:])) #left or right
                #cv2.putText(image, "ForB: {}".format(str(round(action_seq1_mean,2))), (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                #cv2.putText(image, "ForB: {}".format(str(round(action_seq1_mean,2))), (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                #cv2.putText(image, "LorR: {}".format(str(round(action_seq2_mean,2))), (20,150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                #cv2.putText(image, "LorR: {}".format(str(round(action_seq2_mean,2))), (20,150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                           
                if (action_seq1_mean) < -0.8 : #forward 
                    if (action_seq2_mean) < 0.05 :
                        this_action = 'Left_Forward'
                        
                    elif (action_seq2_mean) > 0.25 :
                        this_action = 'Right_Forward'
                    else :
                        this_action = 'Forward'
                      
                elif (action_seq1_mean) > 0.2 : #backward
                    if (action_seq2_mean) < 0.05 :
                        this_action = 'Left_Backward'
                    elif (action_seq2_mean) > 0.21 :
                        this_action = 'Right_Backward'
                    else :
                        this_action = 'Backward'
            
                else : #center
                    if (action_seq2_mean) < 0.05 :
                        this_action = 'Left'
                    elif (action_seq2_mean) > 0.25 :
                        this_action = 'Right'
                    else :
                        this_action = 'Center'
            
                #For TCP (x:-1~1, y:-1~1)      
                tcp_list = []
                y = (action_seq1_mean + 0.3) * -2
                x_forward = (action_seq2_mean - 0.15) * 10
                x_backward = (action_seq2_mean - 0.13) * 12.5
                x_center = (action_seq2_mean - 0.15) * 10
                
                if y > 1 : y = 1
                if y < -1: y = -1
                if x_forward > 1 : x_forward = 1
                if x_forward < -1: x_forward = -1
                if x_backward > 1 : x_backward = 1
                if x_backward < -1: x_backward = -1
                if x_center > 1 : x_center = 1
                if x_center < -1: x_center = -1
                
                tcp_list.append(y)
                if y > 0.5:
                    tcp_list.append(round(x_forward,1))
                elif y < 0.5:
                    tcp_list.append(round(x_backward,1))
                else:
                    tcp_list.append(round(x_center,1)) 
                print(tcp_list)

                cv2.putText(image, "state: {}".format(str(this_action)), (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(image, "state: {}".format(str(this_action)), (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                #cv2.putText(image, "y: {}".format((round(y,1))), (20,200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                #cv2.putText(image, "y: {}".format((round(y,1))), (20,200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                #cv2.putText(image, "x_F: {}".format((round(x_forward,1))), (20,250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                #cv2.putText(image, "x_F: {}".format((round(x_forward,1))), (20,250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                #cv2.putText(image, "x_C: {}".format((round(x_backward,1))), (20,300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                #cv2.putText(image, "x_C: {}".format((round(x_backward,1))), (20,300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                #cv2.putText(image, "x_B: {}".format((round(x_center,1))), (20,350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                #cv2.putText(image, "x_B: {}".format((round(x_center,1))), (20,350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, "o", (100+round(tcp_list[1]*50),150-round(tcp_list[0]*50)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 4, cv2.LINE_AA)
                
                
            else : #right foot in front
            
                this_action = '?'
                action_seq1_mean = 0
                action_seq2_mean = 0
            
                action_seq1.append(nose[2]-r_foot[2]) #forward or backward
                action_seq2.append(r_hip[0]-l_shoulder[0]) #left or right
            
                action_seq1_mean = np.mean(np.array(action_seq1[-5:])) #forward or backward
                action_seq2_mean = np.mean(np.array(action_seq2[-5:])) #left or right
                #cv2.putText(image, "ForB: {}".format(str(round(action_seq1_mean,2))), (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                #cv2.putText(image, "ForB: {}".format(str(round(action_seq1_mean,2))), (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                #cv2.putText(image, "LorR: {}".format(str(round(action_seq2_mean,2))), (20,150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                #cv2.putText(image, "LorR: {}".format(str(round(action_seq2_mean,2))), (20,150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                           
                if (action_seq1_mean) < -0.9 : #forward 
                    if (action_seq2_mean) < -0.25 :
                        this_action = 'Left_Forward'
                        
                    elif (action_seq2_mean) > 0.25 :
                        this_action = 'Right_Forward'
                    else :
                        this_action = 'Forward'
                      
                elif (action_seq1_mean) > -0.1 : #backward
                    if (action_seq2_mean) < -0.21 :
                        this_action = 'Left_Backward'
                    elif (action_seq2_mean) > -0.05 :
                        this_action = 'Right_Backward'
                    else :
                        this_action = 'Backward'
            
                else : #center
                    if (action_seq2_mean) < -0.25 :
                        this_action = 'Left'
                    elif (action_seq2_mean) > -0.05 :
                        this_action = 'Right'
                    else :
                        this_action = 'Center'
            
                #For TCP
                tcp_list = []       
                y = (action_seq1_mean + 0.5) * -2.5
                x_forward = (action_seq2_mean) * 4
                x_backward = (action_seq2_mean + 0.13) * 12.5
                x_center = (action_seq2_mean + 0.15) * 10
                
                if y > 1 : y = 1
                if y < -1: y = -1
                if x_forward > 1 : x_forward = 1
                if x_forward < -1: x_forward = -1
                if x_backward > 1 : x_backward = 1
                if x_backward < -1: x_backward = -1
                if x_center > 1 : x_center = 1
                if x_center < -1: x_center = -1
                
                tcp_list.append(y)
                if y > 0.5:
                    tcp_list.append(round(x_forward,1))
                elif y < 0.5:
                    tcp_list.append(round(x_backward,1))
                else:
                    tcp_list.append(round(x_center,1))
                print(tcp_list) 

                cv2.putText(image, "state: {}".format(str(this_action)), (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(image, "state: {}".format(str(this_action)), (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                #cv2.putText(image, "y: {}".format((round(y,1))), (20,200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                #cv2.putText(image, "y: {}".format((round(y,1))), (20,200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                #cv2.putText(image, "x_F: {}".format((round(x_forward,1))), (20,250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                #cv2.putText(image, "x_F: {}".format((round(x_forward,1))), (20,250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                #cv2.putText(image, "x_C: {}".format((round(x_backward,1))), (20,300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                #cv2.putText(image, "x_C: {}".format((round(x_backward,1))), (20,300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                #cv2.putText(image, "x_B: {}".format((round(x_center,1))), (20,350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                #cv2.putText(image, "x_B: {}".format((round(x_center,1))), (20,350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, "o", (100+round(tcp_list[1]*50),150-round(tcp_list[0]*50)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 4, cv2.LINE_AA)
        
        except:
            pass
        
        cv2.imshow('Sportainment', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
