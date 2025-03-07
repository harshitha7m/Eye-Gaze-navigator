import cv2
import numpy as np
import pyautogui

def detect_eyes(frame, face_cascade, eye_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        detected_eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in detected_eyes:
            eye_center = (x + ex + ew // 2, y + ey + eh // 2)
            eyes.append(eye_center)

    return eyes

# Initialize the video capture and classifiers
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Variables for blinking detection
blink_counter = 0
blink_threshold = 3  # Number of consecutive frames to consider a blink
selection_made = False

# Variables for cursor smoothing
smooth_x, smooth_y = 0, 0
alpha = 0.2  # Smoothing factor

while True:
    ret, frame = cap.read()
    if not ret:
        break

    eyes = detect_eyes(frame, face_cascade, eye_cascade)

    if eyes:
        # Assume the first detected eye is used for cursor control
        eye_center = eyes[0]
        screen_x = np.interp(eye_center[0], [0, frame.shape[1]], [pyautogui.size().width,0 ])
        screen_y = np.interp(eye_center[1], [0, frame.shape[0]], [0, pyautogui.size().height])

        # Apply smoothing
        smooth_x = alpha * screen_x + (1 - alpha) * smooth_x
        smooth_y = alpha * screen_y + (1 - alpha) * smooth_y

        pyautogui.moveTo(smooth_x, smooth_y)

        # Reset blink counter if eyes are detected
        blink_counter = 0
        selection_made = False

    else:
        blink_counter += 1
        if blink_counter > blink_threshold:
            if not selection_made:
                pyautogui.click()
                selection_made = True
                blink_counter = 0  # Reset the blink counter after a click

    cv2.imshow('Eye Tracker', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
