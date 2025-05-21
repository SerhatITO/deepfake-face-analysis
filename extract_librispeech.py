import tarfile
import os

tar_path = "C:/Users/HAZEL/Desktop/train-clean-360.tar.gz"
output_path = "C:/Users/HAZEL/Desktop/LibriSpeech/train-clean-360"

with tarfile.open(tar_path, "r:gz") as tar:
    members = tar.getmembers()
    for member in members:
        target_path = os.path.join(output_path, os.path.relpath(member.name, "LibriSpeech/train-clean-360"))
        if not os.path.exists(target_path):
            try:
                tar.extract(member, path="C:/Users/HAZEL/Desktop/LibriSpeech")
                print(f"Çıkarıldı: {member.name}")
            except Exception as e:
                print(f"Hata: {member.name} -> {e}")
