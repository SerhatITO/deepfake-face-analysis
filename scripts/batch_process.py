import os
import cv2
import dlib
from tqdm import tqdm

# dlib ile yüz dedektörü
detector = dlib.get_frontal_face_detector()

# Girdi ve çıktı klasörleri
INPUT_DIR = "../data/Celeb-DF-v2"
OUTPUT_DIR = "../faces"
FRAME_INTERVAL = 5
MAX_FACES_PER_VIDEO = 10

def is_real(class_name):
    return class_name in ["Celeb-real", "YouTube-real"]

def process_video(video_path, out_dir):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_faces = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_INTERVAL == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            if len(faces) == 0:
                print(f"❌ Yüz bulunamadı: {os.path.basename(video_path)}")
            for i, face in enumerate(faces):
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cropped = frame[y:y+h, x:x+w]
                resized = cv2.resize(cropped, (224, 224))

                base = os.path.splitext(os.path.basename(video_path))[0]
                filename = f"{base}_frame{frame_count:03}_face{i}.jpg"
                filepath = os.path.join(out_dir, filename)

                cv2.imwrite(filepath, resized)
                saved_faces += 1

                if saved_faces >= MAX_FACES_PER_VIDEO:
                    break

        frame_count += 1
        if saved_faces >= MAX_FACES_PER_VIDEO:
            break

    cap.release()

def main():
    os.makedirs(os.path.join(OUTPUT_DIR, "real"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "fake"), exist_ok=True)

    for class_name in os.listdir(INPUT_DIR):
        class_path = os.path.join(INPUT_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        label = "real" if is_real(class_name) else "fake"
        out_dir = os.path.join(OUTPUT_DIR, label)

        print(f"🔄 İşleniyor: {class_name} ({label})")

        for filename in tqdm(os.listdir(class_path)):
            if not filename.endswith(".mp4"):
                continue

            video_path = os.path.join(class_path, filename)
            try:
                process_video(video_path, out_dir)
            except Exception as e:
                print(f"HATA ({filename}): {e}")

if __name__ == "__main__":
    main()
