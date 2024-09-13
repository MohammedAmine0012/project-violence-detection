#video_processing.py
import os
import cv2

def video_to_frames(video_path, output_folder_frames):
    if not os.path.exists(output_folder_frames):
        os.makedirs(output_folder_frames)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            frame_path = os.path.join(output_folder_frames, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)

            frame_count += 1
        else:
            break

    cap.release()

    return output_folder_frames
