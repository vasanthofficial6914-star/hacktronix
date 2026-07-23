import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.GaussianBlur(frame, (5,5), 0)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    kernel = np.ones((5,5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ball_count = 0

for cnt in contours:

    area = cv2.contourArea(cnt)

    if area < 500:
        continue

    perimeter = cv2.arcLength(cnt, True)

    if perimeter == 0:
        continue

    circularity = 4 * np.pi * area / (perimeter * perimeter)

    if circularity > 0.75:

        ball_count += 1

        (x, y), radius = cv2.minEnclosingCircle(cnt)

        center = (int(x), int(y))

        cv2.circle(frame, center, int(radius), (0,255,0), 2)
        
        cv2.putText(frame,
            f"Radius: {int(radius)} px",
            (center[0]-30, center[1]+25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,255,0),
            2)

        cv2.putText(frame,
                    f"Ball {ball_count}",
                    (center[0]-20, center[1]-20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2)
    cv2.putText(frame,
            f"Balls Detected: {ball_count}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2)

    cv2.imshow("Ball Detection", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()