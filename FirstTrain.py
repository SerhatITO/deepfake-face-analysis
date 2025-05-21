import os
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from TTS.api import TTS

# === AYARLAR ===
audio_base_path = r"C:/Users/HAZEL/Desktop/ASVspoof2017_V2_train"
protocol_file = r"C:/Users/HAZEL/Desktop/protocol_V2/ASVspoof2017_V2_train.trn.txt"

X = []
y = []

# === 1. ASVspoof Verisini Yükle ===
with open(protocol_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        filename = parts[0]
        label_str = parts[1]
        label = 1 if label_str == "spoof" else 0

        audio_path = os.path.join(audio_base_path, filename)
        if os.path.exists(audio_path):
            y_audio, sr = librosa.load(audio_path, sr=16000)
            mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc.T, axis=0)
            X.append(mfcc_mean)
            y.append(label)
        else:
            print("Eksik dosya:", audio_path)

# === 2. LibriSpeech'ten Sahte Veri Üret (TTS ile) ===
print("LibriSpeech verisinden sahte sesler üretiliyor...")
librispeech = tfds.load(
    "librispeech",
    split="train_clean100",
    as_supervised=True,
    data_dir="C:/Users/HAZEL/tensorflow_datasets"  # Önceden indirilen veriyi kullan
)

tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
os.makedirs("generated_spoof", exist_ok=True)

for i, (audio, text) in enumerate(tfds.as_numpy(librispeech.take(100))):  # sadece 100 sahte üret
    sentence = text.decode("utf-8")
    spoof_path = f"generated_spoof/spoof_{i}.wav"

    # --- Dosya varsa atla ---
    if os.path.exists(spoof_path):
        print(f"Zaten var: {spoof_path}, atlanıyor.")
    else:
        print(f"Oluşturuluyor: {spoof_path}")
        tts.tts_to_file(text=sentence, file_path=spoof_path)

    # Özellik çıkarımı her durumda yapılmalı (model girdi listesine ekleniyor)
    y_audio, sr = librosa.load(spoof_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    X.append(mfcc_mean)
    y.append(1)  # SPOOF etiketi


print("Sahte ses üretimi tamamlandı.")

# === 3. Veriyi Hazırla ===
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.reshape(-1, 13, 1)
X_test = X_test.reshape(-1, 13, 1)

# === 4. CNN Modeli ===
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(13, 1)),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

def lr_scheduler(epoch, lr):
    return lr if epoch < 10 else lr * 0.5

lr_scheduler_callback = LearningRateScheduler(lr_scheduler)

# === 5. Modeli Eğit ===
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test),
          callbacks=[early_stopping, lr_scheduler_callback])

# === 6. Sonuçları Yazdır ===
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("Doğruluk (Accuracy):", accuracy_score(y_test, y_pred))
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred, target_names=["genuine", "spoof"]))
