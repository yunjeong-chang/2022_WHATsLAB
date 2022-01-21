import cv2 as cv
import numpy as np
import math
import pandas as pd
import mediapipe as mp
import ktb

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def weight_distance(pose1, pose2, conf1):
    sum1 = 1 / np.sum(conf1)
    sum2 = 0
    for i in range(len(pose1)):
        conf_ind = math.floor(i / 3) # each index i has x and y that share same confidence score #original 2, z 3
        sum2 += conf1[conf_ind] * abs(pose1[i] - pose2[i])
    weighted_dist = sum1 * sum2
    return weighted_dist

def similarity_score(pose1, pose2,conf1):
    p1 = []
    p2 = []
    pose_1 = np.array(pose1, dtype=float)
    pose_2 = np.array(pose2, dtype=float)

    # Normalize coordinates
    pose_1[:,0] = pose_1[:,0] / max(pose_1[:,0])
    pose_1[:,1] = pose_1[:,1] / max(pose_1[:,1])
    pose_1[:,2] = pose_1[:,2] / max(pose_1[:,2]) #z
    pose_2[:,0] = pose_2[:,0] / max(pose_2[:,0])
    pose_2[:,1] = pose_2[:,1] / max(pose_2[:,1])
    pose_2[:,2] = pose_2[:,2] / max(pose_2[:,2]) #z

    # Turn (16x2) into (32x1)
    for joint in range(pose_1.shape[0]):
        x1 = pose_1[joint][0]
        y1 = pose_1[joint][1]
        z1 = pose_1[joint][2] #z
        x2 = pose_2[joint][0]
        y2 = pose_2[joint][1]
        z2 = pose_2[joint][2] #z

        p1.append(x1)
        p1.append(y1)
        p1.append(z1) #z
        p2.append(x2)
        p2.append(y2)
        p2.append(z2) #z

    p1 = np.array(p1)
    p2 = np.array(p2)

    scoreB = weight_distance(p1, p2, conf1)

    return scoreB

def get_pose_conf(results):
    pose = []
    conf = []
    for coor in results.pose_landmarks.landmark:
        x = coor.x
        y = coor.y
        #z = coor.z #z
        visibility = coor.visibility
        pose.append((x,y))#z
        conf.append(visibility)
    return pose, conf

def get_pose_conf_pick(results):
    pose = []
    conf = []
    
    pose.append((results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z))
    conf.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility)
    pose.append((results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z))
    conf.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility)
    pose.append((results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].z))
    conf.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].visibility)
    pose.append((results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].z))
    conf.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility)
    pose.append((results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].z))
    conf.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility)
    pose.append((results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].z))
    conf.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility)
    
    return pose, conf

def process_pose(pose1,pose2):
    pose1_new = np.array(pose1)
    pose2_new = np.array(pose2)

    pose1_new[:,0] = pose1_new[:,0] - min(pose1_new[:,0])
    pose1_new[:,1] = pose1_new[:,1] - min(pose1_new[:,1])
    pose1_new[:,2] = pose1_new[:,2] - min(pose1_new[:,2]) #z

    pose2_new[:,0] = pose2_new[:,0] - min(pose2_new[:,0])
    pose2_new[:,1] = pose2_new[:,1] - min(pose2_new[:,1])
    pose2_new[:,2] = pose2_new[:,2] - min(pose2_new[:,2]) #z

    resize_x = max(pose2_new[:,0])/max(pose1_new[:,0])
    resize_y = max(pose2_new[:,1])/max(pose1_new[:,1])
    resize_z = max(pose2_new[:,2])/max(pose1_new[:,2]) #z

    pose1_new[:,0] = pose1_new[:,0] * resize_x
    pose1_new[:,1] = pose1_new[:,1] * resize_y
    pose1_new[:,2] = pose1_new[:,2] * resize_z #z
    
    return pose1_new, pose2_new

def quick_sort(array):
    if len(array) <= 1: return array  
    pivot, tail = array[0], array[1:]
    
    leftSide = [x for x in tail if x <= pivot]
    rightSide = [x for x in tail if x > pivot]
    
    return quick_sort(leftSide) + [pivot] + quick_sort(rightSide)


def tcp_req():
    
    # assign IP with socket lib
    temp_s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    temp_s.connect(("8.8.8.8", 80))
    client_ip = temp_s.getsockname()[0]
    temp_s.close()
    print(type(client_ip))
    
    # assign IP as manual
    # client_ip = "192.168.1.33"
    
    
    serverPort = 9080
    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    req = serverSocket.bind((client_ip, serverPort))
    serverSocket.listen()

    server_socket, address = serverSocket.accept()
    
    print("*"*25)
    print("  my IPv4    : {}".format(client_ip))
    print("  connected  : {}\n  port       : {}".format(address[0], serverPort))
    print("*"*25)

def tcp_send(tmp_list):
    msg = {"Forward":tmp_list[0], "Right":tmp_list[1]}
    msg = json.dumps(msg) + "\r\n"
    b_msg = msg.encode()
    res = server_socket.sendall(b_msg)


#-------------------------------------------main----------------------------------------------------

k = ktb.Kinect()
min_score_class = ""
min_score = 0
x = 0
y = 0

with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2) as pose:

    frame_center = cv.imread("pose-dataset/L-CENTER.jpg") #center
    frame_left = cv.imread("pose-dataset/L-LEFT.jpg") 
    frame_right = cv.imread("pose-dataset/L-RIGHT.jpg") 
    frame_forward = cv.imread("pose-dataset/L-FORWARD.jpg") 
    frame_forward_left = cv.imread("pose-dataset/L-FORWARDLEFT.jpg") 
    frame_forward_right= cv.imread("pose-dataset/L-FORWARDRIGHT.jpg") 
    frame_backward = cv.imread("pose-dataset/L-BACKWARD.jpg") 
    frame_backward_left = cv.imread("pose-dataset/L-BACKWARDLEFT.jpg") 
    frame_backward_right = cv.imread("pose-dataset/L-BACKWARDRIGHT.jpg")
    frame_center2 = cv.imread("pose-dataset/R-CENTER.jpg") #center
    frame_left2 = cv.imread("pose-dataset/R-LEFT.jpg") 
    frame_right2 = cv.imread("pose-dataset/R-RIGHT.jpg") 
    frame_forward2 = cv.imread("pose-dataset/R-FORWARD.jpg") 
    frame_forward_left2 = cv.imread("pose-dataset/R-FORWARDLEFT.jpg") 
    frame_forward_right2= cv.imread("pose-dataset/R-FORWARDRIGHT.jpg") 
    frame_backward2 = cv.imread("pose-dataset/R-BACKWARD.jpg") 
    frame_backward_left2 = cv.imread("pose-dataset/R-BACKWARDLEFT.jpg") 
    frame_backward_right2 = cv.imread("pose-dataset/R-BACKWARDRIGHT.jpg") 
     
    frame_center = cv.cvtColor(frame_center, cv.COLOR_BGR2RGB)   
    frame_left = cv.cvtColor(frame_left, cv.COLOR_BGR2RGB)   
    frame_right = cv.cvtColor(frame_right, cv.COLOR_BGR2RGB) 
    frame_forward = cv.cvtColor(frame_forward, cv.COLOR_BGR2RGB) 
    frame_forward_left = cv.cvtColor(frame_forward_left, cv.COLOR_BGR2RGB) 
    frame_forward_right = cv.cvtColor(frame_forward_right, cv.COLOR_BGR2RGB) 
    frame_backward = cv.cvtColor(frame_backward, cv.COLOR_BGR2RGB) 
    frame_backward_left = cv.cvtColor(frame_backward_left, cv.COLOR_BGR2RGB) 
    frame_backward_right = cv.cvtColor(frame_backward_right, cv.COLOR_BGR2RGB)
    frame_center2 = cv.cvtColor(frame_center2, cv.COLOR_BGR2RGB)   
    frame_left2 = cv.cvtColor(frame_left2, cv.COLOR_BGR2RGB)   
    frame_right2 = cv.cvtColor(frame_right2, cv.COLOR_BGR2RGB) 
    frame_forward2 = cv.cvtColor(frame_forward2, cv.COLOR_BGR2RGB) 
    frame_forward_left2 = cv.cvtColor(frame_forward_left2, cv.COLOR_BGR2RGB) 
    frame_forward_right2 = cv.cvtColor(frame_forward_right2, cv.COLOR_BGR2RGB) 
    frame_backward2 = cv.cvtColor(frame_backward2, cv.COLOR_BGR2RGB) 
    frame_backward_left2 = cv.cvtColor(frame_backward_left2, cv.COLOR_BGR2RGB) 
    frame_backward_right2 = cv.cvtColor(frame_backward_right2, cv.COLOR_BGR2RGB) 
    
    frame_center.flags.writeable = False
    frame_left.flags.writeable = False
    frame_right.flags.writeable = False
    frame_forward.flags.writeable = False
    frame_forward_left.flags.writeable = False
    frame_forward_right.flags.writeable = False
    frame_backward.flags.writeable = False
    frame_backward_left.flags.writeable = False
    frame_backward_right.flags.writeable = False
    frame_center2.flags.writeable = False
    frame_left2.flags.writeable = False
    frame_right2.flags.writeable = False
    frame_forward2.flags.writeable = False
    frame_forward_left2.flags.writeable = False
    frame_forward_right2.flags.writeable = False
    frame_backward2.flags.writeable = False
    frame_backward_left2.flags.writeable = False
    frame_backward_right2.flags.writeable = False
    
    result_center = pose.process(frame_center)
    result_left = pose.process(frame_left)
    result_right = pose.process(frame_right)
    result_forward = pose.process(frame_forward)
    result_forward_left = pose.process(frame_forward_left)
    result_forward_right = pose.process(frame_forward_right)
    result_backward = pose.process(frame_backward)
    result_backward_left = pose.process(frame_backward_left)
    result_backward_right = pose.process(frame_backward_right)
    result_center2 = pose.process(frame_center2)
    result_left2 = pose.process(frame_left2)
    result_right2 = pose.process(frame_right2)
    result_forward2 = pose.process(frame_forward2)
    result_forward_left2 = pose.process(frame_forward_left2)
    result_forward_right2 = pose.process(frame_forward_right2)
    result_backward2 = pose.process(frame_backward2)
    result_backward_left2 = pose.process(frame_backward_left2)
    result_backward_right2 = pose.process(frame_backward_right2)
    
    pose_center,conf_center = get_pose_conf_pick(result_center)
    pose_left,conf_left = get_pose_conf_pick(result_left)
    pose_right,conf_right = get_pose_conf_pick(result_right)
    pose_forward,conf_forward = get_pose_conf_pick(result_forward)
    pose_forward_left,conf_forward_left = get_pose_conf_pick(result_forward_left)
    pose_forward_right,conf_forward_right = get_pose_conf_pick(result_forward_right)
    pose_backward,conf_backward = get_pose_conf_pick(result_backward)
    pose_backward_left,conf_backward_left = get_pose_conf_pick(result_backward_left)
    pose_backward_right,conf_backward_right = get_pose_conf_pick(result_backward_right)
    pose_center2,conf_center2 = get_pose_conf_pick(result_center2)
    pose_left2,conf_left2 = get_pose_conf_pick(result_left2)
    pose_right2,conf_right2 = get_pose_conf_pick(result_right2)
    pose_forward2,conf_forward2 = get_pose_conf_pick(result_forward2)
    pose_forward_left2,conf_forward_left2 = get_pose_conf_pick(result_forward_left2)
    pose_forward_right2,conf_forward_right2 = get_pose_conf_pick(result_forward_right2)
    pose_backward2,conf_backward2 = get_pose_conf_pick(result_backward2)
    pose_backward_left2,conf_backward_left2 = get_pose_conf_pick(result_backward_left2)
    pose_backward_right2,conf_backward_right2 = get_pose_conf_pick(result_backward_right2)
    
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2) as pose:
    
    while True :
        frame = k.get_frame(ktb.RAW_COLOR)
        
        frame = cv.resize(frame, dsize = (0, 0), fx = 0.5, fy = 0.5)
        frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        frame = cv.flip(frame, 1)
        
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = None
        results = pose.process(frame)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)    
        frame.flags.writeable = True

        text_new  = ""
        if results.pose_landmarks == None:
            text_new = "Unable to detect pose"

        else:
            pose_current,conf_current = get_pose_conf_pick(results)

            pose_current_new_center, pose_center_new = process_pose(pose_current,pose_center)
            pose_current_new_left, pose_left_new = process_pose(pose_current,pose_left)
            pose_current_new_right, pose_right_new = process_pose(pose_current,pose_right)
            pose_current_new_forward, pose_forward_new = process_pose(pose_current,pose_forward)
            pose_current_new_forward_left, pose_forward_left_new = process_pose(pose_current,pose_forward_left)
            pose_current_new_forward_right, pose_forward_right_new = process_pose(pose_current,pose_forward_right)
            pose_current_new_backward, pose_backward_new = process_pose(pose_current,pose_backward)
            pose_current_new_backward_left, pose_backward_left_new = process_pose(pose_current,pose_backward_left)
            pose_current_new_backward_right, pose_backward_right_new = process_pose(pose_current,pose_backward_right)
            pose_current_new_center2, pose_center_new2 = process_pose(pose_current,pose_center2)
            pose_current_new_left2, pose_left_new2 = process_pose(pose_current,pose_left2)
            pose_current_new_right2, pose_right_new2 = process_pose(pose_current,pose_right2)
            pose_current_new_forward2, pose_forward_new2 = process_pose(pose_current,pose_forward2)
            pose_current_new_forward_left2, pose_forward_left_new2 = process_pose(pose_current,pose_forward_left2)
            pose_current_new_forward_right2, pose_forward_right_new2 = process_pose(pose_current,pose_forward_right2)
            pose_current_new_backward2, pose_backward_new2 = process_pose(pose_current,pose_backward2)
            pose_current_new_backward_left2, pose_backward_left_new2 = process_pose(pose_current,pose_backward_left2)
            pose_current_new_backward_right2, pose_backward_right_new2 = process_pose(pose_current,pose_backward_right2)

            score_center = similarity_score(pose_current_new_center, pose_center_new, conf_current)
            score_left = similarity_score(pose_current_new_left, pose_left_new, conf_current)
            score_right = similarity_score(pose_current_new_right, pose_right_new, conf_current)
            score_forward = similarity_score(pose_current_new_forward, pose_forward_new, conf_current)
            score_forward_left = similarity_score(pose_current_new_forward_left, pose_forward_left_new, conf_current)
            score_forward_right = similarity_score(pose_current_new_forward_right, pose_forward_right_new, conf_current)
            score_backward = similarity_score(pose_current_new_backward, pose_backward_new, conf_current)
            score_backward_left = similarity_score(pose_current_new_backward_left, pose_backward_left_new, conf_current)
            score_backward_right = similarity_score(pose_current_new_backward_right, pose_backward_right_new, conf_current)
            score_center2 = similarity_score(pose_current_new_center2, pose_center_new2, conf_current)
            score_left2 = similarity_score(pose_current_new_left2, pose_left_new2, conf_current)
            score_right2 = similarity_score(pose_current_new_right2, pose_right_new2, conf_current)
            score_forward2 = similarity_score(pose_current_new_forward2, pose_forward_new2, conf_current)
            score_forward_left2 = similarity_score(pose_current_new_forward_left2, pose_forward_left_new2, conf_current)
            score_forward_right2 = similarity_score(pose_current_new_forward_right2, pose_forward_right_new2, conf_current)
            score_backward2 = similarity_score(pose_current_new_backward2, pose_backward_new2, conf_current)
            score_backward_left2 = similarity_score(pose_current_new_backward_left2, pose_backward_left_new2, conf_current)
            score_backward_right2 = similarity_score(pose_current_new_backward_right2, pose_backward_right_new2, conf_current)
            
            min_score = min([score_center,score_left,score_right,score_forward,score_forward_left,score_forward_right,score_backward,score_backward_left,score_backward_right,
            	 	      score_center2,score_left2,score_right2,score_forward2,score_forward_left2,score_forward_right2,score_backward2,score_backward_left2,score_backward_right2])
            x = 0
            y = 0
            if min_score == score_center or min_score == score_center2:
                min_score_class = "center"
                x = 0
                y = 0
            elif min_score == score_left or min_score == score_left2:
                min_score_class = "left"
                x = -1
                y = 0
            elif min_score == score_right or min_score == score_right2:
                min_score_class = "right"
                x = 1
                y = 0
            elif min_score == score_forward or min_score == score_forward2:
                min_score_class = "forward"
                x = 0
                y = 1
                #if min_score == score_forward and min_score > 0.2:
                #    min_score_class = "center"
                #    x = 0
                #    y = 0
            elif min_score == score_forward_left or min_score == score_forward_left2:
                min_score_class = "forward_left"
                x = -1
                y = 1
            elif min_score == score_forward_right or min_score == score_forward_right2:
                min_score_class = "forward_right"
                x = 1
                y = 1
            elif min_score == score_backward or min_score == score_backward2:
                min_score_class = "backward"
                x = 1
                y = -1
            elif min_score == score_backward_left or min_score == score_backward_left2:
                min_score_class = "backward_left"
                x = -1
                y = -1
            elif min_score == score_backward_right or min_score == score_backward_right2:
                min_score_class = "backward_right"
                x = 1
                y = -1   
        
        cv.putText(frame,text_new, (30, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3, cv.LINE_4)
        cv.putText(frame,min_score_class, (300, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv.LINE_4)
        cv.putText(frame,"score: {}".format(min_score), (300, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv.LINE_4)
        
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv.imshow("Pose Similarity",frame)
        
        # resize & rotate for tutorial
        board_frame = cv.resize(frame, dsize=(1280, 720))
        board_frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
        cv.imshow("board_frame",board_frame)
        # print(np.shape(board_frame))
        
        
        # for rotate
        if (cv.waitKey(10) & 0xFF == ord('q')) or (cv.waitKey(10) & 0xFF == 27):
            break

    cv.destroyAllWindows()
