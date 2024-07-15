import cv2
import mediapipe as mp
import pandas as pd

def extract_pose_keypoints(video_paths):
    # Initialize MediaPipe Pose model
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Prepare to store data
    all_data = []

    # Process each video
    for video_path in video_paths:
        data = []
        cap = cv2.VideoCapture(video_path)

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
                        'exercise': 'squats',
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

        # Convert data to DataFrame for this video
        df = pd.DataFrame(data)
        # Save DataFrame to CSV
        csv_filename = f'{video_path.split(".")[0]}_keypoints.csv'
        df.to_csv(csv_filename, index=False)

        # Append to all_data
        all_data.extend(data)

    # Convert all_data to DataFrame for all videos
    all_df = pd.DataFrame(all_data)
    # Save combined DataFrame to CSV
    all_csv_filename = 'all_videos_keypoints.csv'
    all_df.to_csv(all_csv_filename, index=False)

# List of video paths
video_paths = ['squats1.mp4', 'squats2.mp4', 'squats3.mp4']  # Add more paths as needed

# Call function to extract keypoints from each video
extract_pose_keypoints(video_paths)
