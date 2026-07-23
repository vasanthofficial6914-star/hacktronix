import cv2

# Change this according to your object size
KNOWN_WIDTH = 6.5   # Ball diameter in cm

# This value is calibrated later
FOCAL_LENGTH = 700


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur image
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # Detect circles (ball)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        1.2,
        50,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=200
    )

    if circles is not None:
        circles = circles[0]

        for circle in circles:
            x, y, radius = circle.astype(int)

            # Diameter in pixels
            pixel_width = radius * 2

            # Distance calculation
            distance = (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width

            cv2.circle(frame, (x,y), radius, (0,255,0), 2)

            cv2.putText(
                frame,
                f"Distance: {distance:.2f} cm",
                (x-50, y-20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

    cv2.imshow("Distance Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()