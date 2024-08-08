#Make a requirements file
#INPUT

file=r'YOUR PATH\Video.mp4'

# CODE OUTPUT:  Video File with a sketch outline of human body following the actions of the human in the video (and some text, easily removable)

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt


cap = cv2.VideoCapture(file)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  if cap.isOpened():
    success, image = cap.read()
    height, width, channels = image.shape
   
    frame_width=width#because of the flip
    frame_height = height #because of the flip

result = cv2.VideoWriter('Test_6x.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 55, (frame_width,frame_height))
#fourcc = cv2.VideoWriter_fourcc(*'X264')
#out = cv2.VideoWriter('Test_6x.mp4',fourcc, 52.0, (1920,1080))
#cap = cv2.VideoCapture(0)  #for webcam capture
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      
      break

    
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    
    image.flags.writeable = True
    
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
    

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (255, 255, 255)
    thickness = 3
    image = cv2.putText(image, 'Can Gen-AI create audio instructions? ', org, font, fontScale, color, thickness, cv2.LINE_AA)
    org = (50, 150)
    image = cv2.putText(image, 'And ask about missing info? ', org, font, fontScale, color, thickness, cv2.LINE_AA)
    
    cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1)) # Flip the image horizontally for a selfie-view display.
    x=cv2.flip(image,1)
    print(results.face_landmarks)
    #out.write(x)
    result.write(x)

    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
result.release()  #only 1 is needed, either out or result - which one works
cv2.destroyAllWindows()
