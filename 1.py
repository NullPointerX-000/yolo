from ultralytics import YOLO
import cv2
 
model = YOLO('Yolo-Weights\\yolov8l.pt')
results = model("2.png")
results[0].show()