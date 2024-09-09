import cv2
import dlib
import json
import time
import argparse
from scipy.spatial import distance as dist
from imutils import face_utils

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def main(video_path, frame_skip):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    cap = cv2.VideoCapture(video_path)
    frame_counter = 0
    eye_contact_frames = []
    start_time = time.time()

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_length = frame_count / fps if fps > 0 else 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        if frame_counter % frame_skip != 1:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        for face in faces:
            landmarks = predictor(gray, face)
            landmarks_points = face_utils.shape_to_np(landmarks)

            left_eye = landmarks_points[42:48]
            right_eye = landmarks_points[36:42]
            left_EAR = eye_aspect_ratio(left_eye)
            right_EAR = eye_aspect_ratio(right_eye)

            if left_EAR > 0.2 and right_EAR > 0.2:
                eye_contact_frames.append(frame_counter)

    end_time = time.time()
    processing_time = end_time - start_time

    results = {
        'total_frames': frame_counter,
        'frames_processed': len(eye_contact_frames),
        'eye_contact_frames': eye_contact_frames,
        'eye_contact_frequency': len(eye_contact_frames) / frame_counter if frame_counter > 0 else 0,
        'video_length_seconds': video_length,
        'processing_time_seconds': processing_time,
        'frame_skip': frame_skip
    }

    with open('eye_contact_results.json', 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)

    cap.release()
    cv2.destroyAllWindows()
    print("Analysis complete. Results saved to eye_contact_results.json.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze eye contact in a video.')
    parser.add_argument('video_path', type=str, help='Path to the video file.')
    parser.add_argument('frame_skip', type=int, help='Number of frames to skip between analyses.')
    args = parser.parse_args()
    main(args.video_path, args.frame_skip)
