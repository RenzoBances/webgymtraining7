#!/usr/bin/python

import cv2
import mediapipe as mp
import numpy as np
import time
import streamlit as st
import pickle
import pandas as pd

import Exercises.UpcSystemCost as UpcSystemCost

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose    

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle 


def start(sets, reps, secs, df_trainer_coords, df_trainers_costs):
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    sets_counter = 0 
    stframe = st.empty()
    start = 0
    counter = 0
    resultados_acum = []
    frames_sec = 15
    df_results_coords_total = pd.DataFrame()

    while sets_counter < sets:
        # Squats reps_counter variables
        reps_counter = 0
        stage = None
         # Load Model.

        # Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            cap.isOpened()
            while reps_counter <= reps:
                ret, frame = cap.read()
                
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                # Make detection
                results = pose.process(image)
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                

                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark

                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks]).flatten())

                    # Concate rows
                    row = pose_row

                    # Make Detections
                    X = pd.DataFrame([row])
                    body_language_class = LoadModel().predict(X)[0]
                    body_language_prob = LoadModel().predict_proba(X)[0]
                    body_language_prob1 = body_language_prob*100
                    body_language_prob1=round(body_language_prob1[np.argmax(body_language_prob1)],2)
                    #print(f'class: {body_language_class}, prob: {body_language_prob}')
                    # Get coordinates
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    
                    # Setup status box
                    cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
                    
                    # Rep data
                    cv2.putText(image, 'REPS', (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(reps_counter), 
                                (10,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                    
                    # Stage data
                    cv2.putText(image, 'STAGE', (65,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, stage, 
                                (60,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                    
                    # Calculate angle
                    angle = calculate_angle(hip, knee, ankle)
                    
                    # Visualize angle
                    cv2.putText(image, str(angle), 
                                tuple(np.multiply(knee, [640, 480]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                            ) 

                    # Render detections
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                            )
                    if body_language_prob1 > 70:      
                        if angle > 160:
                                stage = "down"
                        elif angle < 100 and stage =='down':
                                print(f'Paso')
                                stage = "up"
                                time.sleep(1)
                                #funcion de Costos()
                                df_results_coords_total = UpcSystemCost.process(frame_rgb,mp_drawing,mp_pose,results,
                                                                                counter,start,frames_sec,df_trainer_coords,
                                                                                df_trainers_costs,df_results_coords_total,
                                                                                sets_counter,reps_counter)
                                counter +=1
                                start +=1
                                #inicio,c,results,resultados_acum=start_cost(inicio,c,results,resultados_acum)
                                reps_counter +=1
                    else:               
                         stage = ""
                    #cv2.imshow('Mediapipe Feed', image)
                    stframe.image(image,channels = 'BGR',use_column_width=True)

                    # Used to end early
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

                except:
                    pass   
            sets_counter += 1                
            if (sets_counter!=sets):
                try:
                    cv2.putText(image, 'FINISHED SET', (100,250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)
                    #cv2.imshow('Mediapipe Feed', image)
                    stframe.image(image,channels = 'BGR',use_column_width=True)
                    cv2.waitKey(1)
                    time.sleep(secs)   

                except:
                    #cv2.imshow('Mediapipe Feed', image)
                    stframe.image(image,channels = 'BGR',use_column_width=True)
                    pass 
                           
    cv2.rectangle(image, (50,180), (600,400), (0,255,0), -1)
    cv2.putText(image, 'FINISHED EXERCISE', (100,250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(image, 'REST FOR 30s' , (155,350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
    df_results_coords_total.to_csv("./resultados_costos/Squats_resultados_costos.csv",index=False)   
    #cv2.imshow('Mediapipe Feed', image)
    stframe.image(image,channels = 'BGR',use_column_width=True)
    #cv2.waitKey(1) 
    time.sleep(10)          
    cap.release()
    cv2.destroyAllWindows()
    #cv2.destroyAllWindows()

def LoadModel():
    model_weights = './Exercises/model_weights/weights_body_language.pkl'
    with open(model_weights, "rb") as f:
        model = pickle.load(f)
    return model