from ultralytics import YOLO
import cv2
import os
import time

# Change this to the weights that you want to use
model = YOLO("best_20epoch.pt")

classes = list(model.names.values())
res_dict = dict.fromkeys(classes, [])

correct = 0
total = 0

start_time = time.perf_counter()
for classname in classes:
    folder = os.path.join("my_data", classname)
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            results = model(img, conf=0.25, verbose=False)
            best_result = -1
            best_conf = 0
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    confidence = box.conf[0]
                    if confidence > best_conf:
                        best_conf = confidence
                        best_result = class_id
            res_dict[classname].append((best_result, best_conf))
            if best_result != -1 and classname == model.names[best_result]:
                correct += 1
            total += 1
end_time = time.perf_counter()
print(f"Elapsed time: {end_time - start_time}")
print(correct / total)


for classname in classes:
    class_total = 0
    class_correct = 0
    for prediction in res_dict[classname]:
        if prediction[0] != -1 and classname == model.names[prediction[0]]:
            class_correct += 1
        class_total += 1
    print(f"{classname}: {(class_correct / class_total) * 100}%")

# I highly recommend piping the output of the script into a text file
# 20epoch_results.txt, 60epoch_results.txt, or nano_results.txt