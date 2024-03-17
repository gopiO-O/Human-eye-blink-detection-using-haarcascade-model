import cv2
import numpy as np

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    C = np.linalg.norm(eye[0] - eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

EAR_THRESHOLD = 0.2

while cap.isOpened():
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_detector.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]
            eye_aspect_ratio_value = eye_aspect_ratio(eye_roi)

            if eye_aspect_ratio_value < EAR_THRESHOLD:
                print("Alcohol not detected")

            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow("window", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
