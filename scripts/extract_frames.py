import cv2
import os

def extract_frames(video_path, output_folder, frame_rate=10):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_rate == 0:
            filename = os.path.join(output_folder, f"frame_{frame_id:04d}.jpg")
            cv2.imwrite(filename, frame)
            frame_id += 1
        count += 1

    cap.release()
    print(f"{frame_id} frames saved to {output_folder}")
