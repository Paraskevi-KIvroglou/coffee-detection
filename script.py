import cv2
import cv2.data
import numpy as np
import pygame
import os

pygame.init()
# Set up the Pygame window
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption('Pygame Window')

pygame.mixer.init()
sound = pygame.mixer.Sound("./sms-alert-4-daniel_simon.wav")

haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if os.path.exists(haar_cascade_path):
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)
    print("Haar cascade loaded successfully.")
else:
    print(f"Haar cascade file not found: {haar_cascade_path}")

cap = cv2.VideoCapture(0) 

is_drinking = False

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3,5)

    for (x,y, w, h) in faces:
        mouth_roi = gray[y + h //2 : y + h, x: x + w]
        _, mouth_threshold = cv2.threshold(mouth_roi, 50,225, cv2.THRESH_BINARY)
        white_pixels = np.sum(mouth_threshold == 225)

        if white_pixels > 10000:
            if not is_drinking:
                print("Drinking Coffee detected!")
                is_drinking = True
                sound.play()
        else:
            is_drinking = False

        cv2.rectangle(frame, (x,y), (x + w, y + h), (255,0,0), 2)

    cv2.imshow("frame",frame)