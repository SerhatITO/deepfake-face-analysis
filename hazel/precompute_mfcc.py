import os
import librosa
import numpy as np

# Ana ses klasörü (alt klasörlerle birlikte taranacak)
audio_base_dir = r"C:\Users\HAZEL\Desktop\archive"

# MFCC kayıt klasörü
mfcc_output_dir = r"C:\Users\HAZEL\Desktop\mfcc_archive"
os.makedirs(mfcc_output_dir, exist_ok=True)

n_mfcc = 40

for root, dirs, files in os.walk(audio_base_dir):
    for file in files:
        if file.lower().endswith(".wav"):
            audio_path = os.path.join(root, file)
            try:
                y, sr = librosa.load(audio_path, sr=None)
                if sr != 16000:
                    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                    sr = 16000
                
                min_len = sr * 1  # minimum 1 saniye padding
                if len(y) < min_len:
                    y = np.pad(y, (0, min_len - len(y)), mode='constant')
                
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T

                # Dosyanın audio_base_dir altındaki göreceli yolu
                rel_path = os.path.relpath(audio_path, audio_base_dir)
                # Yol ayracı yerine _ koyarak güvenli dosya adı oluştur
                safe_name = os.path.splitext(rel_path)[0].replace(os.sep, "_") + ".npy"

                save_path = os.path.join(mfcc_output_dir, safe_name)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, mfcc)

                print(f"✅ {rel_path} -> {safe_name} kaydedildi.")
            except Exception as e:
                print(f"❌ Hata: {audio_path} - {e}")
