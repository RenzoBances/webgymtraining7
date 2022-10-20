# 1.1. PYTHON LIBRARIES
#######################
import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import time
import base64
from random import randrange
import pandas as pd
import main
#1.2. OWN LIBRARIES
###################
#simport main
import Exercises.SquatsUPC
#import Exercises.Extensions
#import Exercises.Crunches
#import Exercises.Rows
#import Exercises.BenchPress

import Exercises.UpcSystemCost as UpcSystemCost
import Exercises.UpcAngleCostSquats as UpcAngleCostSquats


# 2. FUNCTIONS
##############
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def print_system_angle(frame):
    reps_counter = str(0)
    stage = "down"

    #frame, reps_counter, stage = Exercises.SquatsUPC.start(frame, n_sets,n_reps, seconds_rest_time)
    #Exercises.SquatsUPC.start(frame, n_sets, n_reps, seconds_rest_time)


# 3. HTML CODE
#############
st.set_page_config(
    page_title="STARTER TRAINING - UPC",
    page_icon ="img/upc_logo.png",
)

img_upc = get_base64_of_bin_file('img/upc_logo_50x50.png')
fontProgress = get_base64_of_bin_file('fonts/ProgressPersonalUse-EaJdz.ttf')

st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {{
        width: 336px;        
    }}
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {{
        width: 336px;
        margin-left: -336px;
        background-color: #00F;
    }}
    [data-testid="stVerticalBlock"] {{
        flex: 0;
        gap: 0;
    }}
    #starter-training{{
        padding: 0;
    }}
    #div-upc{{
        #border: 1px solid #DDDDDD;
        background-image: url("data:image/png;base64,{img_upc}");
        position: fixed !important;
        right: 14px;
        bottom: 14px;
        width: 50px;
        height: 50px;
        background-size: 50px 50px;
    }}
    @font-face {{
        font-family: ProgressFont;
        src: url("data:image/png;base64,{fontProgress}");
    }}
    .css-10trblm{{
        color: #FFF;
        font-size: 40px;
        font-family: ProgressFont;    
    }}
    .main {{
        background: linear-gradient(135deg,#a8e73d,#09e7db,#092de7);
        background-size: 180% 180%;
        animation: gradient-animation 3s ease infinite;
        }}

        @keyframes gradient-animation {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
    .block-container{{
        max-width: 100%;
    }}
    </style>
    """ ,
    unsafe_allow_html=True,
)


# 4. PYTHON CODE
#############

if 'camera' not in st.session_state:
    st.session_state['camera'] = 0

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

st.title('STARTER TRAINING')
st.markdown("<div id='div-upc'></span>", unsafe_allow_html=True)

app_mode = st.sidebar.selectbox('Choose your training:',
    ['🏠HOME','Squats','Abs', 'Lunges', 'Push Up', 'Glute Bridge']
)

#id_trainer = randrange(3) + 1
id_trainer = 4

reik=0

exercise_to_do = {}

if app_mode =='🏠HOME':
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("**POSE_LANDMARKS**<br>Una lista de puntos de referencia de la pose. Cada punto de referencia consta de lo siguiente:<br><ul><li><b>X & Y:</b> coordenadas de referencia normalizadas a [0.0, 1.0] por el ancho y la altura de la imagen, respectivamente.</li><li><b>Z:</b> Representa la profundidad del punto de referencia con la profundidad en el punto medio de las caderas como origen, y cuanto menor sea el valor, más cerca estará el punto de referencia de la cámara. La magnitud de z usa aproximadamente la misma escala que x.</li><li><b>Visibilidad:</b> un valor en [0.0, 1.0] que indica la probabilidad de que el punto de referencia sea visible (presente y no ocluido) en la imagen.</li></ul><br>",
        unsafe_allow_html=True)
    st.markdown("**MODELO DE PUNTOS DE REFERENCIA DE POSE (BlazePose GHUM 3D)**<br>El modelo de puntos de referencia en MediaPipe Pose predice la ubicación de 33 puntos de referencia de pose (consulte la figura a continuación).<br>",
        unsafe_allow_html=True)
    st.image("img/pose_landmarks_model.png", width=600)

elif app_mode =='Squats':
    st.sidebar.markdown('---')
    st.sidebar.markdown('**SQUATS**')

    vista_exercises = randrange(3) + 1
    vista_dict = {1: "frontal", 2: "lateral", 3: "tres-cuartos"}
    vistal_text = vista_dict[vista_exercises]
    vista_gif = 'img/vista_' + vistal_text + '.gif'

    st.sidebar.markdown("**- Distancia cámara-usuario :** 1 metro", unsafe_allow_html=True)
    st.sidebar.markdown("**- Vista corporal requerida :** " + vistal_text, unsafe_allow_html=True)
    st.sidebar.image(vista_gif, width=150)

    st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
    n_sets = st.sidebar.number_input("Sets", min_value=1, max_value=10, value=3)
    n_reps = st.sidebar.number_input("Reps", min_value=1, max_value=12, value=10)
    seconds_rest_time = st.sidebar.number_input("Rest Time (seconds)", min_value=1, max_value=60, value=10)
    exercise_to_do[app_mode] = {"reps":n_reps,"sets":n_sets,"secs":seconds_rest_time}
    st.sidebar.markdown("<hr/>", unsafe_allow_html=True)

    cam_button, cam_status = st.sidebar.columns(2)

    with cam_button:
        webcam = st.button("Webcam")
        
    
    video_trainer_file="videos_trainer/Squats/Squats"+str(id_trainer)+".mp4"
    df_trainer_coords = pd.read_csv("videos_trainer/Squats/Puntos_Squats"+str(id_trainer)+".csv")

    del df_trainer_coords['segundo']
    df_trainers_costs = pd.read_csv("videos_trainer/Squats/Costos_Squats_promedio.csv")

    trainer, user = st.columns(2)
    with st.spinner('Starting in 5 seconds...'):
                time.sleep(5)
    with trainer:        
        st.markdown("**TRAINER**", unsafe_allow_html=True)
        st.video(video_trainer_file, format="video/mp4", start_time=0)
        
    with user:
        st.markdown("**USER**", unsafe_allow_html=True)



        if(webcam):
            video_capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)

            # Cámara apagada
            if st.session_state['camera'] % 2 != 0:
                with cam_status:
                    #st.text(str(st.session_state['camera']) + ": Impar-apagado")
                    st.warning('Apagada', icon="⚠️")
                    st.session_state['camera'] += 1

                video_capture.release()
            # Cámara encendida
            else:                
                with cam_status:                    
                    #st.text(str(st.session_state['camera']) + ": Par-encendido")
                    st.success('Encendida', icon="✅")
                    st.session_state['camera'] += 1
                
                stframe = st.empty()
                main.start(exercise_to_do, df_trainer_coords, df_trainers_costs)
            video_capture.release()
            cv2.destroyAllWindows()

    
    
    st.markdown("<hr/>", unsafe_allow_html=True)


elif app_mode =='Abs':
    a=0
elif app_mode =='Lunges':
    a=0
elif app_mode =='Push Up':
    a=0
elif app_mode =='Glute Bridge':
    a=0
else:
    a=0

