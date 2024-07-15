import cv2
import mediapipe as mp
import pandas as pd

# Initialize MediaPipe Pose model
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to process video and extract keypoints
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    data = []

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the BGR image to RGB.
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Process the image and detect pose
            results = pose.process(image)

            if results.pose_landmarks:
                # Extract landmarks
                landmarks = results.pose_landmarks.landmark

                # Extract desired key points
                key_points = {
                    'exercise': 'pushup',
                    'wrist_left_x': landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    'wrist_left_y': landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                    'wrist_right_x': landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                    'wrist_right_y': landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                    'elbow_left_x': landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    'elbow_left_y': landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                    'elbow_right_x': landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                    'elbow_right_y': landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                    'shoulder_left_x': landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    'shoulder_left_y': landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                    'shoulder_right_x': landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    'shoulder_right_y': landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                    'hip_left_x': landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    'hip_left_y': landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                    'hip_right_x': landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    'hip_right_y': landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                    'knee_left_x': landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    'knee_left_y': landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                    'knee_right_x': landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                    'knee_right_y': landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                    'ankle_left_x': landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                    'ankle_left_y': landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                    'ankle_right_x': landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                    'ankle_right_y': landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
                }

                data.append(key_points)

    cap.release()
    return data

# List of video paths to process
video_paths = ['1.mp4', '2.mp4', '3.mp4', '4.mp4', '5.mp4', '6.mp4', '7.mp4', '8.mp4', '9.mp4', '10.mp4', '11.mp4', 'pushup.mp4']

# Process each video and save data
all_data = []
for path in video_paths:
    video_data = process_video(path)
    all_data.extend(video_data)

# Convert data to DataFrame
df = pd.DataFrame(all_data)
df.to_csv('pushup_keypoints.csv', index=False)
