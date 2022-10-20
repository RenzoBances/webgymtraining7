#!/usr/bin/python

import cv2
import numpy as np
import datetime
import pandas as pd

def dp(dist_mat):
    """
    Find minimum-cost path through matrix `dist_mat` using dynamic programming.

    The cost of a path is defined as the sum of the matrix entries on that
    path. See the following for details of the algorithm:

    - http://en.wikipedia.org/wiki/Dynamic_time_warping
    - https://www.ee.columbia.edu/~dpwe/resources/matlab/dtw/dp.m

    The notation in the first reference was followed, while Dan Ellis's code
    (second reference) was used to check for correctness. Returns a list of
    path indices and the cost matrix.
    """

    N, M = dist_mat.shape
    
    # Initialize the cost matrix
    cost_mat = np.zeros((N + 1, M + 1))
    for i in range(1, N + 1):
        cost_mat[i, 0] = np.inf
    for i in range(1, M + 1):
        cost_mat[0, i] = np.inf

    # Fill the cost matrix while keeping traceback information
    traceback_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            penalty = [
                cost_mat[i, j],      # match (0)
                cost_mat[i, j + 1],  # insertion (1)
                cost_mat[i + 1, j]]  # deletion (2)
            i_penalty = np.argmin(penalty)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
            traceback_mat[i, j] = i_penalty

    # Traceback from bottom right
    i = N - 1
    j = M - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        tb_type = traceback_mat[i, j]
        if tb_type == 0:
            # Match
            i = i - 1
            j = j - 1
        elif tb_type == 1:
            # Insertion
            i = i - 1
        elif tb_type == 2:
            # Deletion
            j = j - 1
        path.append((i, j))

    # Strip infinity edges from cost_mat before returning
    cost_mat = cost_mat[1:, 1:]
    return (path[::-1], cost_mat)

def calculate_costs(user_array, df_trainer_coords, start, df_trainers_costs):
    
    results_costs = []
    results_index = []
    results_costs_al = []
    results_costs_al_normalized = []
    trainer_array = []

    for i in df_trainer_coords.columns:
        ct = datetime.datetime.now()
        print(str(ct) + " Evaluating the second: " + str(start))
        trainer_array.append(df_trainer_coords[i][start])

    x = np.array(user_array) 
    y = np.array(trainer_array)

    N = x.shape[0]
    M = y.shape[0]
    dist_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            dist_mat[i, j] = abs(x[i] - y[j])

        # DTW
    path, cost_mat = dp(dist_mat)

    x_path, y_path = zip(*path)
    results_index.append(start)
    results_costs_al.append(cost_mat[N - 1, M - 1])
    results_costs_al_normalized.append(cost_mat[N - 1, M - 1]/(N + M))
    results_costs.append([start,cost_mat[N - 1, M - 1],cost_mat[N - 1, M - 1]/(N + M)])
    start += 1

    return results_costs

def validate_costs(results_costs, start, df_trainers_costs):
    eval_sec = str(start)
    starting_cost = str(round(df_trainers_costs.Costo_alineamiento[start]-df_trainers_costs.Desviacion_estandar[start], 2))
    final_cost = str(round(df_trainers_costs.Costo_alineamiento[start]+df_trainers_costs.Desviacion_estandar[start], 2))
    resulting_cost = str(round(results_costs[0][1], 2))
    
    if (results_costs[0][1] <= df_trainers_costs.Costo_alineamiento[start] - df_trainers_costs.Desviacion_estandar[start] or 
        results_costs[0][1] <= df_trainers_costs.Costo_alineamiento[start] + df_trainers_costs.Desviacion_estandar[start]): # promedio +- desviación estandar (para evitar casos rápidos o lentos)

        message_validation = "Posicion correcta"
        color_validation = (255, 0, 0)
        start += 1

    else:
        message_validation = "Posicion incorrecta"
        color_validation = (0, 0, 255)

   
    return start, eval_sec, starting_cost, final_cost, resulting_cost, message_validation, color_validation
    
def print_system_cost(frame, results_frame, mp_drawing, mp_pose, eval_sec, starting_cost, final_cost, resulting_cost, message_validation, color_validation):
    #0. Setting landmarks: Estableciendo los puntos de referencia de pose
    if results_frame.pose_landmarks is not None:
        mp_drawing.draw_landmarks(
            frame,
            results_frame.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(
                color=(255, 0, 160),
                thickness=2,
                circle_radius=3),
                mp_drawing.DrawingSpec(
                    color=(255,255,255),
                    thickness=2
                    ))
    #1. Esquina superior izquierda: Evaluación de costos trainer vs user
    cv2.rectangle(frame, (0,0), (270,50), (245,117,16), -1)
    cv2.putText(frame, 
                "Evaluando segundo: "+ eval_sec, #Título
                (5,20),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (255,255,255),
                1, 
                cv2.LINE_AA)
    cv2.putText(frame, 
                "Rango costos: [" + starting_cost + " - " + final_cost + "]", #Rango costos
                (5,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,255,255),
                1, 
                cv2.LINE_AA)

    
    #2. Esquina superior derecha: Posición correcta/incorrecta
    cv2.rectangle(frame, (460,0), (650,50), (255,255,255), -1)
    cv2.putText(frame, 
                "Costo usuario: " + resulting_cost, #Costo resultante 
                (465,20),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5,
                color_validation,
                1, 
                cv2.LINE_AA)
    cv2.putText(frame, 
                message_validation, #Posición correcta/incorrecta
                (465,40),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5,
                color_validation,
                1, 
                cv2.LINE_AA)

def process(frame, mp_drawing, mp_pose, results_frame, counter, start, frames_sec, df_trainer_coords, df_trainers_costs, df_results_coords_total, sets_counter, reps_counter):
    #df_results_coords_total = pd.DataFrame()
    results_array = []
    for i in range(0, len(results_frame.pose_landmarks.landmark)):
        results_array.append(results_frame.pose_landmarks.landmark[i].x)
        results_array.append(results_frame.pose_landmarks.landmark[i].y)
        results_array.append(results_frame.pose_landmarks.landmark[i].z)
        results_array.append(results_frame.pose_landmarks.landmark[i].visibility)
    
    user_array = results_array
    
    #UpcSystemCost
    results_costs = calculate_costs(user_array, df_trainer_coords, start, df_trainers_costs)
    
    start, eval_sec, starting_cost, final_cost, resulting_cost, message_validation, color_validation = validate_costs(results_costs, start, df_trainers_costs)
    
    print_system_cost(frame, results_frame, mp_drawing, mp_pose, eval_sec, starting_cost, final_cost, resulting_cost, message_validation, color_validation)

    results_array.append(starting_cost)
    results_array.append(final_cost)
    results_array.append(resulting_cost)
    results_array.append(message_validation)
    results_array.append(sets_counter)
    results_array.append(reps_counter)
    
    df_results_coords = pd.DataFrame(np.reshape(results_array, (138, 1)).T)
    df_results_coords['segundo'] = str(counter/frames_sec)
    
    if counter == 0:
        df_results_coords_total = df_results_coords.copy()
    else:
        df_results_coords_total = pd.concat([df_results_coords_total, df_results_coords])

    return df_results_coords_total