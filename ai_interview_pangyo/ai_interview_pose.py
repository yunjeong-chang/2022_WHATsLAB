import cv2
import os
import numpy as np
import mediapipe as mp 
import ktb
import timeit

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
    cv2.putText(image, x, coord, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

action_1_list = []
action_2_list = []
action_3_list = []
action_4_list = []
action_5_list = []
action_6_list = []
cnt_1 = 0
cnt_2 = 0
cnt_3 = 0
cnt_4 = 0
cnt_5 = 0
cnt_6 = 0
cnt_1_warning = 0
cnt_2_warning = 0
cnt_3_warning = 0
cnt_4_warning = 0
cnt_5_warning = 0
cnt_6_warning = 0
time_1 = 0
time_2 = 0
time_3 = 0
time_4 = 0
time_5 = 0
time_6 = 0
time_1_list = []
time_2_list = []
time_3_list = []
time_4_list = []
time_5_list = []
time_6_list = []

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
            
            l_ear = pose_landmarks(landmarks, 'LEFT_EAR')
            r_ear = pose_landmarks(landmarks, 'RIGHT_EAR')
            l_mouth = pose_landmarks(landmarks, 'MOUTH_LEFT')
            r_mouth = pose_landmarks(landmarks, 'MOUTH_RIGHT')
            l_index = pose_landmarks(landmarks, 'LEFT_INDEX')
            r_index = pose_landmarks(landmarks, 'RIGHT_INDEX')
            l_shoulder = pose_landmarks(landmarks, 'LEFT_SHOULDER')
            r_shoulder = pose_landmarks(landmarks, 'RIGHT_SHOULDER')
            l_hip = pose_landmarks(landmarks, 'LEFT_HIP')
            r_hip = pose_landmarks(landmarks, 'RIGHT_HIP')
            l_knee = pose_landmarks(landmarks, 'LEFT_KNEE')
            r_knee = pose_landmarks(landmarks, 'RIGHT_KNEE')
            l_ankle = pose_landmarks(landmarks, 'LEFT_ANKLE')
            r_ankle = pose_landmarks(landmarks, 'RIGHT_ANKLE')
            
            angle_head_l = calculate_angle(l_ear, l_hip, l_shoulder)
            angle_head_r = calculate_angle(r_ear, r_hip, r_shoulder)
            angle_body_l = calculate_angle(l_shoulder, l_hip, l_knee)
            angle_body_r = calculate_angle(r_shoulder, r_hip, r_knee)
            
            l_index_shoulder_y = l_index[1] - l_shoulder[1]
            r_index_shoulder_y = r_index[1] - r_shoulder[1]
            l_index_hip_y = l_index[1] - l_hip[1]
            r_index_hip_y = r_index[1] - r_hip[1]
            l_index_hip_x = l_index[0] - l_hip[0]
            r_index_hip_x = r_index[0] - r_hip[0]
            l_shoulder_hip_mean_y = (l_shoulder[1]+l_hip[1])/2
            r_shoulder_hip_mean_y = (r_shoulder[1]+r_hip[1])/2
            
            action_list = []

            # 1 : hand_to_face (손이 얼굴 높이로)
            if l_index_shoulder_y < 0 or r_index_shoulder_y < 0 : 
                action_list.append("hand_to_face")
                action_1_list.append(1) 
                time_1 += 1 #유지시간 카운트
            else :
                action_1_list.append(0)
                if action_1_list[-2] == 1 :
                    cnt_1 += 1 #해당 동작 실행 횟수 카운트
                    time_1_list.append(time_1) #유지시간 저장
                    time_1 = 0
        
            # 2 : tilted_head (기울어진 고개)    
            if angle_head_l > 24 or angle_head_r > 24 : 
                action_list.append("tilted_head")
                action_2_list.append(2) 
                time_2 += 1 #유지시간 카운트
            else :
                action_2_list.append(0)
                if action_2_list[-2] == 2 :
                    time_2_list.append(time_2) #유지시간 저장
                    if time_2 < 100 :
                        cnt_2 += 1 #해당 동작 2초 미만 실행 횟수 카운트
                    else :
                        cnt_2_warning += 1 #해당 동작 2초 이상 실행 횟수 카운트
                    time_2 = 0
            
            # 3 : hand_on_waist (손 허리 짚기)
            if (l_shoulder_hip_mean_y < l_index[1] and l_index[1] < l_hip[1]) or (r_shoulder_hip_mean_y < r_index[1] and r_index[1] < r_hip[1]) :
                if (0 < l_index_hip_x and l_index_hip_x < 0.1) or (0 > r_index_hip_x and r_index_hip_x > -0.1) :
                    action_list.append("hand_on_waist")
                    action_3_list.append(3) 
                    time_3 += 1 #유지시간 카운트
            else :
                action_3_list.append(0)
                if action_3_list[-2] == 3 :
                    time_3_list.append(time_3) #유지시간 저장
                    if time_3 < 100 :
                        cnt_3 += 1 #해당 동작 2초 미만 실행 횟수 카운트
                    else :
                        cnt_3_warning += 1 #해당 동작 2초 이상 실행 횟수 카운트
                    time_3 = 0  
                
            # 4 : hand_on_chest (손 가슴 높이, 양 손 팔짱)
            if (l_shoulder[1] < l_index[1] and l_index[1] < l_shoulder_hip_mean_y) or (r_shoulder[1] < r_index[1] and r_index[1] < r_shoulder_hip_mean_y) :
                if (l_index_hip_x < 0) or (r_index_hip_x > 0) : 
                    action_list.append("hand_on_chest")
                    action_4_list.append(4) 
                    time_4 += 1 #유지시간 카운트
            else :
                action_4_list.append(0)
                if action_4_list[-2] == 4 :
                    time_4_list.append(time_4) #유지시간 저장
                    if time_4 < 100 :
                        cnt_4 += 1 #해당 동작 2초 미만 실행 횟수 카운트
                    else :
                        cnt_4_warning += 1 #해당 동작 2초 이상 실행 횟수 카운트
                    time_4 = 0  
                
            # 5 : tilted_body (기울어진 상체, 허리 스트레칭)
            if angle_body_l < 155 or angle_body_r < 155 : 
                action_list.append("tilted_body")
                action_5_list.append(5) 
                time_5 += 1 #유지시간 카운트
            else :
                action_5_list.append(0)
                if action_5_list[-2] == 5 :
                    cnt_5 += 1 #해당 동작 실행 횟수 카운트
                    time_5_list.append(time_5) #유지시간 저장
                    time_5 = 0
                
            # 6 : normal_posture (배 위에 양 손 모아 공손 or 차렷) 
            if ((l_shoulder_hip_mean_y < l_index[1] and l_index[1] < l_hip[1]+0.02) and (r_shoulder_hip_mean_y < r_index[1] and r_index[1] < r_hip[1]+0.02)) or ((l_hip[1] < l_index[1]) and (r_hip[1] < r_index[1])):
                if ((l_index_hip_x < 0) and (r_index_hip_x > 0)) or ((0.01 < l_index_hip_x and l_index_hip_x < 0.1) and (-0.01 > r_index_hip_x and r_index_hip_x > -0.1)) : 
                    action_list.append("normal_posture")
                    action_6_list.append(6) 
                    time_6 += 1 #유지시간 카운트
            
            else :
                action_6_list.append(0)
                if action_6_list[-2] == 6 :
                    cnt_6 += 1 #해당 동작 유지하지 못한 횟수 카운트
                    time_6_list.append(time_6) #유지시간 저장
                    time_6 = 0  
                 
            cv2.putText(image, "state: {}".format(str(action_list)), (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, "state: {}".format(str(action_list)), (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)    
            #putText(mask, "state: {} ".format(str(action_list)), (20,50))
            #putText(mask, "l_index_hip_y: {} ".format(str(l_index_hip_y)), (20,100))
            #putText(mask, "r_index_hip_y: {} ".format(str(r_index_hip_y)), (20,150))
            #putText(mask, "l_index_hip_x: {} ".format(str(l_index_hip_x)), (20,200))
            #putText(mask, "r_index_hip_x: {} ".format(str(r_index_hip_x)), (20,250))
            
        except:
            pass
          
            
        cv2.imshow('Pose Estimation', image)
        #cv2.imshow('Pose Estimation coord', mask)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    #cap.release()
    cv2.destroyAllWindows()

print('hand_to_face : ', cnt_1)
print('tilted_head: ',cnt_2, cnt_2_warning)
print('hand_on_waist: ',cnt_3, cnt_3_warning)
print('hand_on_chest: ',cnt_4, cnt_4_warning)
print('tilted_body: ', cnt_5)
print('normal_posture: ', cnt_6, time_6_list)

#전송할 목록
#1.손 얼굴 높이   : 해당 동작을 취한 횟수 => cnt1(5회 이상 습관성 피드백)
#2.기울어진 고개  : 해당 동작을 2초 미만 취한 횟수 => cnt2(5회 이상 습관성 피드백) , 해당 동작을 2초 이상 취한 횟수 => cnt2_warning(1회 이상 경고 피드백)
#3.손 허리        : 해당 동작을 2초 미만 취한 횟수 => cnt3, 해당 동작을 2초 이상 취한 횟수 => cnt3_warning
#4.손 팔짱        : 해당 동작을 2초 미만 취한 횟수 => cnt4, 해당 동작을 2초 이상 취한 횟수 => cnt4_warning
#5.기울어진 상체  : 해당 동작을 취한 횟수 => cnt5(10회 이상 습관성 피드백)
#6.정자세(배 위에 양 손 모아 공손,손 차렷) : 해당 동작을 유지하지 못한 횟수 => cnt6, 해당 동작을 유지한 시간 => time_6_list




