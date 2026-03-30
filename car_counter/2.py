import cv2
import cvzone
import math
import numpy as np
from ultralytics import YOLO
from sort import Sort

# 1. Setup & Constants
VIDEO_PATH = r"video\final video.mp4"
MODEL_PATH = r"Yolo-Weights\yolov8l.pt"

TARGET_CLASSES = {"car", "truck", "bus", "motorbike"}

# Detection Line [x1, y1, x2, y2]
LIMITS = [150, 450, 550, 450]

model = YOLO(MODEL_PATH)
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
cap = cv2.VideoCapture(VIDEO_PATH)

total_count = set()  # Use a set for faster O(1) lookups

while cap.isOpened():
    success, img = cap.read()
    if not success: break

   

    # 2. Run Inference
    results = model(img, stream=True)
    detections = []

    for r in results:
        for box in r.boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls_name = model.names[int(box.cls[0])] # Get name directly from model
            
            if cls_name in TARGET_CLASSES and conf > 0.3:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append([x1, y1, x2, y2, conf])

    # 3. Update Tracker
    results_tracker = tracker.update(np.array(detections) if detections else np.empty((0, 5)))
    
    # Draw counting line
    cv2.line(img, (LIMITS[0], LIMITS[1]), (LIMITS[2], LIMITS[3]), (0, 0, 255), 5)

    for x1, y1, x2, y2, obj_id in results_tracker.astype(int):
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2 # Center Point

        # Visuals
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'ID: {obj_id}', (x1, y1 - 10), scale=1, thickness=1, offset=3)
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # 4. Counting Logic
        # 1. Calculate Center
        cx, cy = x1 + w // 2, y1 + h // 2 
        
        # 2. Draw the center dot (CRITICAL for debugging)
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # 3. Flexible Counting Logic
        # Instead of a specific line, we check if the car is in a broad 'box'
        # across the middle of your visible road.
        
        # X-range: horizontal spread of the road
        # Y-range: vertical 'hit zone'
        if 150 < cx < 600 and 400 < cy < 500:
            if obj_id not in total_count:
                total_count.add(obj_id)
                # Visual feedback: Change circle color when counted
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    # Display Count
    cv2.putText(img, str(len(total_count)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 7)
    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()