import cv2
import cvzone
import math
import numpy as np
from ultralytics import YOLO
from sort import Sort

VIDEO_PATH = r"video\final video.mp4"
MODEL_PATH = r"Yolo-Weights\yolov8l.pt"
TARGET_CLASSES = {"car", "truck", "bus", "motorbike"}

# --- NEW: Visual Counting Zone ---
# We will draw this box on screen. If a car's center dot doesn't go 
# INSIDE this blue box, it won't be counted. Adjust these numbers!
ZONE_X1, ZONE_Y1 = 450, 400
ZONE_X2, ZONE_Y2 = 1000, 550 

model = YOLO(MODEL_PATH)
# Lowered min_hits to 1 so cars get IDs immediately
tracker = Sort(max_age=20, min_hits=1, iou_threshold=0.3)
cap = cv2.VideoCapture(VIDEO_PATH)

total_count = set() 

while cap.isOpened():
    success, img = cap.read()
    if not success: break

    results = model(img, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        for box in r.boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls_name = model.names[int(box.cls[0])]
            
            # Lowered confidence temporarily to catch blurry/fast cars
            if cls_name in TARGET_CLASSES and conf > 0.2: 
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

    results_tracker = tracker.update(detections)
    
    # --- DRAW THE COUNTING ZONE (Blue Box) ---
    cv2.rectangle(img, (ZONE_X1, ZONE_Y1), (ZONE_X2, ZONE_Y2), (255, 0, 0), 3)
    cv2.putText(img, "COUNT ZONE", (ZONE_X1, ZONE_Y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)

    for result in results_tracker:
        x1, y1, x2, y2, obj_id = map(int, result)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2 

        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        # Draw the center point
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Check if the center point is inside our Blue Box
        if ZONE_X1 < cx < ZONE_X2 and ZONE_Y1 < cy < ZONE_Y2:
            if obj_id not in total_count:
                total_count.add(obj_id)
                # Flash the box green when a car is counted
                cv2.rectangle(img, (ZONE_X1, ZONE_Y1), (ZONE_X2, ZONE_Y2), (0, 255, 0), 5)
                print(f"Counted ID {obj_id}. Total: {len(total_count)}")

    cv2.putText(img, f"Total: {len(total_count)}", (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 7)
    cv2.imshow("Image", img)
    
    # Slow down the video slightly so you can see what's happening
    if cv2.waitKey(30) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()