from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("best.pt")

# Open the default webcam (0). Try 1 or 2 if you have multiple cameras.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Couldn't open webcam — check permissions or try a different index")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    # Run detection. conf=0.5 means only show predictions it's >=50% sure about.
    results = model(frame, conf=0.25, verbose=False)

    # Draw the boxes + labels onto the frame
    annotated = results[0].plot()

    cv2.imshow("ASL Detector — press Q to quit", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()