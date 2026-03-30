from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture("video/ppe-2-1.mp4") 

model = YOLO("best.pt")
classNames = model.names 

while True:
    print("1. Reading frame...")
    success, img = cap.read()
    
    if not success or img is None:
        print("ERROR: Video ended or path is wrong!")
        break
        
    print("2. Running YOLO model (this might freeze on the first frame)...")
    results = model(img, stream=True)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            
            if conf > 0.5:
                if currentClass in ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']:
                    myColor = (0, 0, 255) 
                elif currentClass in ['Hardhat', 'Safety Vest', 'Mask']:
                    myColor = (0, 255, 0) 
                else:
                    myColor = (255, 0, 0) 

                cvzone.putTextRect(img, f'{currentClass} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                   colorT=(255, 255, 255), colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    print("3. Drawing image to screen!")
    cv2.imshow("image", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()