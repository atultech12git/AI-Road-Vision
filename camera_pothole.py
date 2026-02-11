import tensorflow as tf
import cv2
import numpy as np
import winsound

# ===============================
# LOAD TRAINED MODEL
# ===============================
model = tf.keras.models.load_model(
    "pothole_binary_model.h5",
    compile=False
)

cap = cv2.VideoCapture(0)

# 🔥 Tuned for 0.95 accuracy
POTHOLE_THRESHOLD = 0.90
PLAIN_THRESHOLD = 0.35
CONFIRM_FRAMES = 10


pothole_count = 0
pothole_active = False

print("Press Q to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    score = model.predict(img, verbose=0)[0][0]

    # ===============================
    # DECISION LOGIC
    # ===============================
    if score > POTHOLE_THRESHOLD:
        pothole_count += 1
    elif score < PLAIN_THRESHOLD:
        pothole_count = 0
        pothole_active = False

    if pothole_count >= CONFIRM_FRAMES:
        pothole_active = True

    # ===============================
    # OUTPUT ONLY WHEN POTHOLE
    # ===============================
    if pothole_active:
        cv2.putText(
            frame, "POTHOLE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0, 0, 255), 2
        )
        winsound.Beep(1200, 150)

    cv2.imshow("Pothole Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()