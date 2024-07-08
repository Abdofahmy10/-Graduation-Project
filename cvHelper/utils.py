import mediapipe as mp
import pickle
import cv2
import pandas as pd
import numpy as np 
from cvHelper.landmarks import landmark

mp_holistic = mp.solutions.holistic

def load_model(model_file):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model('models\detect.pkl')

def extract_landmarks(frame):
    # Recolor frame
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make detections
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        results = holistic.process(image)

    # Initialize empty rows for different landmarks
    right_row = []
    face_row = []
    left_row = []
    pose_row = []

    if results.right_hand_landmarks:
        right = results.right_hand_landmarks.landmark
        right_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right]).flatten())
    else:
        # Fill right_row with zeros if right hand landmarks are not detected
        right_row = [0] * 84

    if results.face_landmarks:
        face = results.face_landmarks.landmark
        face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
    else:
        # Fill face_row with zeros if face landmarks are not detected
        face_row = [0] * 1872

    if results.left_hand_landmarks:
        left = results.left_hand_landmarks.landmark
        left_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left]).flatten())
    else:
        # Fill left_row with zeros if left hand landmarks are not detected
        left_row = [0] * 84 

    if results.pose_landmarks:
        pose = results.pose_landmarks.landmark
        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
    else:
        # Fill pose_row with zeros if pose landmarks are not detected
        pose_row = [0] * 132 

    # Concatenate rows
    row = right_row + face_row + pose_row + left_row 

    return row

def predict_photo(image):
    # Extract row of landmarks
    row = extract_landmarks(image)

    X = pd.DataFrame([row], columns=landmark)

    # Make prediction
    body_language_class = model.predict(X)[0]
    body_language_prob = model.predict_proba(X)[0]

    return body_language_class, body_language_prob

def calculate_score(results):
    good_count = sum(1 for result in results if result["body_language_class"] == "good")
    bad_count = sum(1 for result in results if result["body_language_class"] == "bad")
    total = good_count + bad_count
    if total == 0:
        return 0
    return (good_count / total) * 100



