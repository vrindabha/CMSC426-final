import cv2, os

LETTER = "Z"  # ← change this for each letter (skip J and Z!)
SAVE_DIR = f"my_data/{LETTER}"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0
print(f"Capturing for letter {LETTER}. SPACE = save, Q = quit")
while True:
    ret, frame = cap.read()
    if not ret: break
    # counter overlay so you can see how many you've taken
    cv2.putText(frame, f"{LETTER}: {count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Capture - SPACE to save, Q to quit", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        path = f"{SAVE_DIR}/{LETTER}_{count:03d}.jpg"
        cv2.imwrite(path, frame)
        print(f"saved {path}")
        count += 1
    elif key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()