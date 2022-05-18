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
    
capture_list = []
fortune_cookie_cnt = 0
toss_coin_list = []
    
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
            
            l_index = pose_landmarks(landmarks, 'LEFT_INDEX')
            r_index = pose_landmarks(landmarks, 'RIGHT_INDEX')
            
            this_action = ""
            
            #fortune_cookie
            if r_index[2] < -0.7 or l_index[2] < -0.7: #손 앞으로 
                fortune_cookie_cnt += 1 #fortune_cookie start
                if fortune_cookie_cnt == 20 :
                    this_action = "fortune_cookie" 
                    fortune_cookie_cnt = 0 
            else :
                fortune_cookie_cnt = 0 #fortune_cookie end

            #capture & toss_coin
            if r_index[0] > 0.5 : #오른손, 화면 좌측 
                if r_index[1] < 0.4 and r_index[0] > 0.7: 
                    capture_list.append(-1) #capture start(왼쪽에서 시작)
                    toss_coin_list.append(0)
                    
            elif r_index[0] < 0.5 : #오른손, 화면 우측 
                if r_index[1] < 0.4 and r_index[0] < 0.3:
                    capture_list.append(1) #capture end(오른쪽에서 끝)
                    if capture_list[-1] == 1 and capture_list[-2] == -1 : 
                        this_action = "capture"
                    
                elif r_index[1] > 0.4 :
                    toss_coin_list.append(-1) #toss_coin start(아래)
                elif r_index[1] < 0.1 :
                    toss_coin_list.append(1) #toss_coin end(위) 
                    if toss_coin_list[-1] == 1 and toss_coin_list[-2] == -1 :
                        this_action = "toss_coin"              
                    
            if l_index[0] > 0.5 : #왼손, 화면 좌측
                if l_index[1] > 0.4 :
                    toss_coin_list.append(-1) #toss_coin start(아래)
                elif l_index[1] < 0.1 :
                    toss_coin_list.append(1) #toss_coin end(위)
                    if toss_coin_list[-1] == 1 and toss_coin_list[-2] == -1 :
                        this_action = "toss_coin" 
                
            #capture 할 때 toss-coin이 맞물리는 경우가 있는데, 분수대에서 동전 던지는 상황 아니면 toss_coin 나와도 무시하면 될 듯
            
            putText(image, "state: {} ".format(str(this_action)), (20,50))  
            putText(image, "l_index: {} ".format(str([round(l_index[0],1),round(l_index[1],1),round(l_index[2],1)])), (20,100)) 
            putText(image, "r_index: {} ".format(str([round(r_index[0],1),round(r_index[1],1),round(r_index[2],1)])), (20,150)) 
            putText(image, "fortune_cookie_cnt: {} ".format(str(fortune_cookie_cnt)), (20,200)) 
                 
            
        except:
            pass
              
        cv2.imshow('Metaverse', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    #cap.release()
    cv2.destroyAllWindows()




