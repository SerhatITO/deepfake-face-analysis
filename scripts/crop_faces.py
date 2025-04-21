import cv2
import os
from pathlib import Path

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def crop_faces(input_folder, output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for img_file in Path(input_folder).glob("*.jpg"):
        img = cv2.imread(str(img_file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for i, (x, y, w, h) in enumerate(faces):
            face = img[y:y+h, x:x+w]
            out_path = Path(output_folder) / f"{img_file.stem}_face{i}.jpg"
            cv2.imwrite(str(out_path), face)

    print(f"✅ Yüzler kırpıldı ve kaydedildi: {output_folder}")
