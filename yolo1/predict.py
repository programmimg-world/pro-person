import os
from ultralytics import YOLO
import cv2

# Define the video path
video_path =r"C:\yolo_project\yolo1\data\images\test\pexels_videos_2084066 (1080p).mp4"
video_path_out = video_path.replace('.mp4', '_out.mp4')

# Initialize video capture
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape

# Initialize video writer
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# Define the model path
model_path = r'C:\yolo_project\runs\detect\train26\weights\last.pt'
# Load the YOLO model
model = YOLO(model_path)

# Set the threshold for detection
threshold = 0.5

while ret:
    # Perform inference
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Write the frame to the output video
    out.write(frame)

    # Read the next frame
    ret, frame = cap.read()

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
