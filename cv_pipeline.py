import cv2
import numpy as np
import pandas as pd
import time
from ultralytics import YOLO

video = cv2.VideoCapture('sample_video.mp4')
model = YOLO('yolov8n.pt')

fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(
    'output_video.mp4', fourcc, fps, (width, height))

metrics_log = []
frame_count = 0


def classical_processing(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh


while True:
    success, frame = video.read()

    if not success:
        break

    frame_count += 1

    frame_start = time.time()
    results = model(frame, verbose=False)
    frame_end = time.time()
    if frame_count == 1:
        classical_result = classical_processing(frame)
        cv2.imwrite('classical_frame1.png', classical_result)
    person_count = 0
    confidences = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        if class_name == "person":
            person_count += 1
        confidences.append(confidence)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        avg_confidence = sum(confidences) / \
            len(confidences) if confidences else 0
    frame_time = frame_end - frame_start
    fps = 1 / frame_time if frame_time > 0 else 0

    metrics_log.append({
        'frame': frame_count,
        'people_detected': person_count,
        'avg_confidence': round(avg_confidence, 3),
        'processing_time_sec': round(frame_time, 4),
        'fps': round(fps, 2)
    })

    output_video.write(frame)

print("Total frames processed:", frame_count)

video.release()
output_video.release()

metrics_df = pd.DataFrame(metrics_log)
metrics_df.to_csv('frame_metrics.csv', index=False)

print(metrics_df.describe())
