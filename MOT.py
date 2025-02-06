import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model
model = YOLO("yolo11m.pt")

# Without tracking
# video_path = ('people.mp4')
# video_path_out = '{}_out2.mp4'.format(video_path)

# cap = cv2.VideoCapture(video_path)
# ret, frame = cap.read()
# H, W, _ = frame.shape
# out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# threshold = 0.5

# while ret:

#     results = model(frame)[0]

#     for result in results.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = result

#         if score > threshold:
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
#             cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

#     out.write(frame)
#     ret, frame = cap.read()

# cap.release()
# out.release()
# cv2.destroyAllWindows()

########################################################

# Initialize DeepSORT tracker
deep_sort_tracker = DeepSort(max_age=2000) 

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    H, W, _ = frame.shape
    out = cv2.VideoWriter(video_path+"deepSort_out", cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO for object detection
        results = model(frame,classes=0)
        detections = []
        
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box.cpu().numpy()
                if conf > 0.7:  # Confidence threshold
                    detections.append([[x1, y1, x2, y2], conf, int(cls)])
        
        # Update DeepSORT tracker
        tracks = deep_sort_tracker.update_tracks(detections, frame=frame)
        
        for track in tracks:
            if track.is_confirmed():
                x1, y1, x2, y2 = track.to_tlbr()
                track_id = track.track_id
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'ID {track_id}', (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display result
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) == ord('q'):
            break

        out.write(frame)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Run tracking on a sample video
process_video('people.mp4')

########################################################

# Bot Sort
# results = model.track(source="people.mp4", persist=True, save=True)

########################################################

# Byte Track
# results = model.track("people.mp4", persist=True, show=True, tracker="bytetrack.yaml", save=True)
# results = model.track("people.mp4", persist=True, show=True, classes=0, tracker="bytetrack.yaml", save=True)
