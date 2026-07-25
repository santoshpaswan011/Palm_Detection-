import cv2
import mediapipe as mp
import time

# ---- Model path ----
MODEL_PATH = "hand_landmarker.task"

# ---- Tasks API Setup ----
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

# ---- Hand connections ----
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# ---- Shared result ----
latest_result = None

def save_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

# ---- Options ----
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
    result_callback=save_result
)

# ---- Draw landmarks ----
def draw_hand(img, hand_landmarks, w, h):
    for start, end in HAND_CONNECTIONS:
        x1 = int(hand_landmarks[start].x * w)
        y1 = int(hand_landmarks[start].y * h)
        x2 = int(hand_landmarks[end].x * w)
        y2 = int(hand_landmarks[end].y * h)
        cv2.line(img, (x1, y1), (x2, y2), (0, 200, 200), 2)

    for idx, lm in enumerate(hand_landmarks):
        cx, cy = int(lm.x * w), int(lm.y * h)
        if idx in [4, 8, 12, 16, 20]:
            cv2.circle(img, (cx, cy), 8, (0, 0, 255), cv2.FILLED)
        else:
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

# ---- Camera Setup ----
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW for Windows
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

# Warm up camera (flush initial empty frames)
print("🔄 Warming up camera...")
for _ in range(10):
    cap.read()
print("✅ Camera ready!")

pTime = 0

# ---- Main Loop ----
with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, img = cap.read()

        if not success or img is None:
            print("❌ Failed to grab frame, retrying...")
            time.sleep(0.1)
            continue          # ← retry instead of break

        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)

        if latest_result and latest_result.hand_landmarks:
            for i, hand_landmarks in enumerate(latest_result.hand_landmarks):

                draw_hand(img, hand_landmarks, w, h)

                xList = [int(lm.x * w) for lm in hand_landmarks]
                yList = [int(lm.y * h) for lm in hand_landmarks]
                xMin = max(0, min(xList) - 20)
                xMax = min(w, max(xList) + 20)
                yMin = max(0, min(yList) - 20)
                yMax = min(h, max(yList) + 20)
                cv2.rectangle(img, (xMin, yMin), (xMax, yMax), (0, 255, 0), 2)

                if latest_result.handedness:
                    label = latest_result.handedness[i][0].display_name
                    cv2.putText(img, f'{label} Hand', (xMin, yMin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                wx = int(hand_landmarks[0].x * w)
                wy = int(hand_landmarks[0].y * h)
                cv2.circle(img, (wx, wy), 12, (0, 255, 255), cv2.FILLED)

        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Palm Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()