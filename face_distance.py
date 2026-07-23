import cv2
import mediapipe as mp

# Initialize Face Detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

cap = cv2.VideoCapture(0)

# -------- Calibration --------
KNOWN_DISTANCE = 50      # cm (measure this once)
KNOWN_FACE_WIDTH = 16    # Average face width in cm
FOCAL_LENGTH = 650       # Adjust this if needed
# -----------------------------

while True:
    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if results.detections:
        for detection in results.detections:

            bbox = detection.location_data.relative_bounding_box

            h, w, _ = frame.shape

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            # Draw face box
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

            # Estimate distance
            distance = (KNOWN_FACE_WIDTH * FOCAL_LENGTH) / bw

            # Show distance
            cv2.putText(frame,
                        f"Distance: {distance:.1f} cm",
                        (x, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2)

    cv2.imshow("Face Distance Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()