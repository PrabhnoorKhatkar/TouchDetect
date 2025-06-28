import cv2
import mediapipe as mp
import time
import numpy as np
import screeninfo

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

touch_start_time = None
TOUCH_SECONDS = 3  # Set to 3-5 as needed
blanked = False
black_window_open = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hand = hands.process(rgb)
    results_face = face_detection.process(rgb)

    left_hand_detected = False
    face_box = None

    # Detect face and get bounding box
    if results_face.detections:
        for detection in results_face.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x1, y1 = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
            w, h = int(bboxC.width * iw), int(bboxC.height * ih)
            face_box = (x1, y1, x1 + w, y1 + h)
            break  # Only use the first detected face

    # Detect left hand and check proximity to face
    touching = False
    if results_hand.multi_handedness and results_hand.multi_hand_landmarks and face_box:
        for idx, hand_handedness in enumerate(results_hand.multi_handedness):
            label = hand_handedness.classification[0].label
            if label == 'Left':
                left_hand_detected = True
                hand_landmarks = results_hand.multi_hand_landmarks[idx]
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    if face_box[0] <= x <= face_box[2] and face_box[1] <= y <= face_box[3]:
                        touching = True
                        break

    current_time = time.time()
    if touching:
        if touch_start_time is None:
            touch_start_time = current_time
        elapsed = current_time - touch_start_time
        if elapsed >= TOUCH_SECONDS:
            blanked = True
        else:
            blanked = False
    else:
        touch_start_time = None
        blanked = False

    # Show/hide black overlay window
    if blanked and not black_window_open:
        screen = screeninfo.get_monitors()[0]
        black = np.zeros((screen.height, screen.width, 3), dtype=np.uint8)
        # Add text to the center
        text1 = "Stop that to continue"
        text2 = "Press ESC to quit"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        thickness = 6
        # First line
        text_size1, _ = cv2.getTextSize(text1, font, font_scale, thickness)
        text_x1 = (screen.width - text_size1[0]) // 2
        text_y1 = (screen.height // 2) - 20
        cv2.putText(black, text1, (text_x1, text_y1), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)
        # Second line
        text_size2, _ = cv2.getTextSize(text2, font, font_scale // 2, thickness // 2)
        text_x2 = (screen.width - text_size2[0]) // 2
        text_y2 = text_y1 + 60
        cv2.putText(black, text2, (text_x2, text_y2), font, font_scale // 2, (200,200,200), thickness // 2, cv2.LINE_AA)
        cv2.namedWindow("BLANK", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("BLANK", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("BLANK", black)
        black_window_open = True
    elif not blanked and black_window_open:
        cv2.destroyWindow("BLANK")
        black_window_open = False

    # No camera window shown (runs in background)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()