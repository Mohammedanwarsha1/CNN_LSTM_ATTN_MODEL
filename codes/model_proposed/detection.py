import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
from layers import SelfAttention  # Import SelfAttention from layers.py

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def preprocess_frame(frame):
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Perform pose estimation
    results = pose.process(frame_rgb)
    
    if results.pose_landmarks:
        # Extract keypoints
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        return np.array(keypoints)
    else:
        return None

def extract_keypoints_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        keypoints = preprocess_frame(frame)
        if keypoints is not None:
            keypoints_list.append(keypoints)
    
    cap.release()
    return np.array(keypoints_list)

# Example usage
video_path = 'E:\yoga\Yoga-Pose-Classification-and-Skeletonization\encodingHumanActivity\squat_demo.mp4'
keypoints_data = extract_keypoints_from_video(video_path)

# Ensure the keypoints data has the correct shape
keypoints_data = np.expand_dims(keypoints_data, axis=2)  # Add the channel dimension

# Load the trained model
model_filename = 'E:\yoga\Yoga-Pose-Classification-and-Skeletonization\encodingHumanActivity\model_with_self_attn_LOTO_results\best_model_with_self_attn_squat_exercise_fold_9.h5'
model = keras.models.load_model(model_filename, custom_objects={'SelfAttention': SelfAttention})

# Make predictions
predictions = model.predict(keypoints_data, batch_size=16)

# Decode predictions (assuming one-hot encoding)
predicted_labels = np.argmax(predictions, axis=1)

# Print the predicted labels
print(predicted_labels)