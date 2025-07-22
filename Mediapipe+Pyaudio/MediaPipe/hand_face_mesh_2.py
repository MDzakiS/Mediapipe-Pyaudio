import cv2
import mediapipe as mp
import pandas as pd
import datetime

# --- Input nama file dan label di awal ---
filename_input = input("Masukkan nama file CSV (tanpa .csv): ").strip()
label = input("Masukkan label untuk data ini (misal: happy, angry, neutral): ").strip()

# --- Face landmark ID yang dipertahankan ---
face_ids = [
    0, 13, 14, 17,
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    33, 246, 161, 160, 159, 158, 157, 173,
    263, 466, 388, 387, 386, 385, 384, 398,
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300
]

# --- Inisialisasi MediaPipe ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

pose = mp_pose.Pose()
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
all_data = []

print("[INFO] Tekan 's' untuk menyimpan frame, 'q' untuk keluar.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_results = pose.process(rgb)
    hand_results = hands.process(rgb)
    face_results = face_mesh.process(rgb)

    frame_data = []

    # Pose (siku kiri & kanan = ID 13 dan 14)
    if pose_results.pose_landmarks:
        for i in [13, 14]:
            lm = pose_results.pose_landmarks.landmark[i]
            frame_data.extend([lm.x, lm.y, lm.z, lm.visibility])
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Hand
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                frame_data.extend([lm.x, lm.y, lm.z])
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Face
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            for idx in face_ids:
                lm = face_landmarks.landmark[idx]
                frame_data.extend([lm.x, lm.y, lm.z])
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
            )

    # Tampilkan instruksi
    cv2.putText(frame, "Tekan 's' untuk simpan, 'q' untuk keluar",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Gesture + Mulut Capture", frame)
    key = cv2.waitKey(1)

    if key == ord('s') and len(frame_data) > 0:
        frame_data.insert(0, label)
        all_data.append(frame_data.copy())
        print("[INFO] Frame disimpan.")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Simpan ke CSV
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"{filename_input}_{label}_{timestamp}.csv"
pd.DataFrame(all_data).to_csv(filename, index=False)
print(f"[INFO] Data disimpan ke: {filename}")
