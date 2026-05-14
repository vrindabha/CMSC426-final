from ultralytics import YOLO
import time
import cv2

# Load your trained model
model = YOLO("best.pt")

# Open the default webcam (0). Try 1 or 2 if you have multiple cameras.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Couldn't open webcam — check permissions or try a different index")

current_sign = -1
timer_start = 0
captured = False
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    # Run detection. conf=0.25 means only show predictions it's >=25% sure about.
    
    results = model(frame, conf=0.25, verbose=False)
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            confidence = box.conf[0]
            if confidence >= .25:
                if current_sign == class_id:
                    if not captured and time.perf_counter_ns() - timer_start >= 750000000:
                        print(class_name)
                        captured = True

                else:
                    timer_start = time.perf_counter_ns()
                    captured = False
                    current_sign = class_id
            else:
                timer_start = time.perf_counter_ns()
                captured = False
                current_sign = -1
    # Draw the boxes + labels onto the frame
    annotated = results[0].plot()

    cv2.imshow("ASL Detector — press Q to quit", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()