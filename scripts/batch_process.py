import os
from pathlib import Path
from scripts.extract_frames import extract_frames
from scripts.crop_faces import crop_faces
from scripts.resize_faces import resize_faces

# Ayarlar
REAL_VIDEOS_PATHS = [
    "data/Celeb-DF-v2/Celeb-real",
    "data/Celeb-DF-v2/YouTube-real"
]
FAKE_VIDEOS_PATH = "data/Celeb-DF-v2/Celeb-synthesis"

FRAME_RATE = 5
RESIZE_DIM = (224, 224)

def process_all(videos_path, label):
    for video_path in Path(videos_path).glob("*.mp4"):
        name = video_path.stem
        print(f"\n▶ İşleniyor: {name}")

        # 1. Frame çıkar
        frames_output = Path(f"extracted_frames/{label}/{name}")
        extract_frames(str(video_path), str(frames_output), frame_rate=FRAME_RATE)

        # 2. Yüz kırp
        faces_output = Path(f"faces/{label}/{name}")
        crop_faces(str(frames_output), str(faces_output))

        # 3. Resize et
        resized_output = Path(f"faces/{label}_resized/{name}")
        resize_faces(str(faces_output), str(resized_output), size=RESIZE_DIM)

if __name__ == "__main__":
    print("🚀 Gerçek videolar işleniyor...")
    for real_path in REAL_VIDEOS_PATHS:
        process_all(real_path, "real")

    print("\n⚠️ Sahte videolar işleniyor...")
    process_all(FAKE_VIDEOS_PATH, "fake")

    print("\n✅ Tüm veriler işlendi.")
