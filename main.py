# lower half detection instead of nose because impacted by hair shadow

import cv2
import numpy as np
from termcolor import colored

net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
cap = cv2.VideoCapture(1)
max_intensity = 255
min_intensity = 0
darkness_threshold = 60
black_detected = False

while True:
    _, img = cap.read()
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 2)

            print(colored(f"Face detected at coordinates: {(x, y, x1, y1)} with confidence: {confidence}", 'green'))

            w = x1 - x
            h = y1 - y
            lower_half_face_region = img[y + h // 2: y + h, x: x + w]

            cv2.rectangle(img, (x, y + h // 2), (x + w, y + h), (255, 0, 0), 2)

            avg_intensity = np.mean(lower_half_face_region)
            darkness_complexity = ((max_intensity - avg_intensity) / (max_intensity - min_intensity)) * 100
            text = f'Darkness: {darkness_complexity:.2f}%'
            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

            print(colored(f"Darkness complexity: {darkness_complexity}", 'blue'))

            if darkness_complexity >= darkness_threshold:
                black_detected = True
                print(colored('Black detected!', 'red'))

    if black_detected:
        text = 'BLACK DETECTED'
        cv2.putText(img, text, (img.shape[1] // 2, img.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        black_detected = False
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
