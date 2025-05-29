import os
import re
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve

# === GPU Ayarı === #
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{len(gpus)} GPU bulundu.")
    except RuntimeError as e:
        print(e)
else:
    print("GPU bulunamadı.")
print("Kullanılabilir GPU cihazları:", gpus)

# === Dağıtım Stratejisi === #
strategy = tf.distribute.MirroredStrategy()
print("\U0001f9e0 Eğitim stratejisi:", strategy)

# === AYARLAR === #
csv_files = [
    "avspoof_labeled_train_dev_normalized.csv",
    "La_dev_labels.csv",
    "La_eval_labels.csv",
    "La_train_labels.csv",
    "labeled_audio.csv",
    "LJSpeech-1.1_labeled.csv",
    "spoof_labels.csv"
]

audio_base_dir = r"C:\\Users\\HAZEL\\Desktop\\VoiceProcess2"
mfcc_cache_dir = r"C:\\Users\\HAZEL\\Desktop\\mfcc_cache"
checkpoint_dir = "checkpoints1"
model_save_path = "saved_model1.keras"
results_dir = "results1"

batch_size = 32
epochs = 40
n_mfcc = 40

os.makedirs(results_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(mfcc_cache_dir, exist_ok=True)

# === CSV Verilerini Yükle === #
all_data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
all_data = all_data.dropna(subset=["path"])
all_data = all_data[all_data["path"].apply(lambda x: isinstance(x, str) and x.lower() != 'nan' and x.strip() != '')]
print(f"Temizlenmiş toplam örnek: {len(all_data)}")

# === Train/Test Split === #
train_df, test_df = train_test_split(all_data, test_size=0.2, stratify=all_data["label"], random_state=42)

# === MFCC Hesaplama === #
def compute_mfcc(audio_path):
    try:
        y, sr_orig = librosa.load(audio_path, sr=None)
        if sr_orig != 16000:
            y = librosa.resample(y, orig_sr=sr_orig, target_sr=16000)
            sr = 16000
        else:
            sr = sr_orig
        if len(y) < sr:
            y = np.pad(y, (0, sr - len(y)))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T
        return mfcc.astype(np.float32)
    except Exception as e:
        print(f"MFCC hesaplama hatası: {audio_path} - {e}")
        return None

# === MFCC Yolu ve Okuma === #
def get_mfcc_path(audio_path):
    filename = os.path.basename(audio_path)
    return os.path.join(mfcc_cache_dir, filename + ".npy")

def preprocess_mfcc(audio_path):
    mfcc_path = get_mfcc_path(audio_path)
    if os.path.exists(mfcc_path):
        try:
            return np.load(mfcc_path).astype(np.float32)
        except Exception as e:
            print(f"⚠️ Bozuk MFCC dosyası atlandı: {mfcc_path} - {e}")
            return None
    else:
        # MFCC cache yok, hesapla
        mfcc = compute_mfcc(audio_path)
        if mfcc is not None:
            try:
                np.save(mfcc_path, mfcc)
                print(f"✅ MFCC hesaplandı ve kaydedildi: {mfcc_path}")
            except Exception as e:
                print(f"⚠️ MFCC kaydetme hatası: {mfcc_path} - {e}")
            return mfcc
        else:
            print(f"⚠️ MFCC hesaplama başarısız: {audio_path}")
            return None


# === Veri Üretici === #
def data_generator(df):
    for _, row in df.iterrows():
        mfcc = preprocess_mfcc(row["path"])
        if mfcc is None:
            continue
        label = int(row["label"])
        yield mfcc, label

# === Dataset Oluşturma === #
def get_dataset(df):
    output_types = (tf.float32, tf.int32)
    output_shapes = ((None, n_mfcc), ())
    dataset = tf.data.Dataset.from_generator(lambda: data_generator(df), output_types, output_shapes)
    dataset = dataset.map(lambda x, y: (x, tf.one_hot(y, 2)))
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None, n_mfcc], [2]))
    dataset = dataset.repeat()
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    return dataset.with_options(options).prefetch(tf.data.AUTOTUNE)

train_ds = get_dataset(train_df)
test_ds = get_dataset(test_df)

steps_per_epoch = (len(train_df) + batch_size - 1) // batch_size
validation_steps = (len(test_df) + batch_size - 1) // batch_size

# === Model Tanımı === #
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Masking(mask_value=0., input_shape=(None, n_mfcc)),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    return model

# === Model Eğitimi === #
with strategy.scope():
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')]
    )

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "model-epoch{epoch:02d}.keras"),
        save_weights_only=False,
        save_best_only=False,
        verbose=1
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',      # İzlenecek metrik
    patience=5,              # Kaç epoch boyunca iyileşme beklenmeli
    restore_best_weights=True,  # En iyi epoch'un ağırlıklarını geri yükle
    verbose=1
)


    keras_checkpoints = [fname for fname in os.listdir(checkpoint_dir) if re.match(r"model-epoch(\d+)\.keras", fname)]
    if keras_checkpoints:
        epochs_found = [int(re.findall(r"model-epoch(\d+)\.keras", fname)[0]) for fname in keras_checkpoints]
        latest_epoch = max(epochs_found)
        latest_ckpt = os.path.join(checkpoint_dir, f"model-epoch{latest_epoch:02d}.keras")
        print(f"Checkpoint yüklendi: {latest_ckpt}")
        model = tf.keras.models.load_model(latest_ckpt)
        initial_epoch = latest_epoch
    else:
        print("Checkpoint bulunamadı, eğitim sıfırdan başlıyor.")
        initial_epoch = 0

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=[checkpoint_cb, early_stopping_cb],
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )

# === Modeli Kaydet === #
model.save(model_save_path)

# === Değerlendirme === #
loss, acc, precision, recall, auc_score = model.evaluate(test_ds, steps=validation_steps)

# F1 Skoru
y_true, y_pred, y_score = [], [], []
for x_batch, y_batch in test_ds.take(validation_steps):
    preds = model.predict(x_batch)
    y_true.extend(tf.argmax(y_batch, axis=1).numpy())
    y_pred.extend(tf.argmax(preds, axis=1).numpy())
    y_score.extend(preds[:, 1])

f1 = f1_score(y_true, y_pred)

metrics = {
    "Test Loss": loss,
    "Test Accuracy": acc,
    "Test Precision": precision,
    "Test Recall": recall,
    "Test AUC": auc_score,
    "Test F1 Score": f1
}

with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
    for k, v in metrics.items():
        f.write(f"{k}: {v}\n")

# === Eğitim Grafikleri === #
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.legend()
plt.title('Loss')

plt.subplot(2, 2, 2)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.legend()
plt.title('Accuracy')

plt.subplot(2, 2, 3)
plt.plot(history.history['precision'], label='Eğitim Precision')
plt.plot(history.history['val_precision'], label='Doğrulama Precision')
plt.legend()
plt.title('Precision')

plt.subplot(2, 2, 4)
plt.plot(history.history['recall'], label='Eğitim Recall')
plt.plot(history.history['val_recall'], label='Doğrulama Recall')
plt.legend()
plt.title('Recall')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "training_metrics.png"))
plt.close()

# === Karışıklık Matrisi === #
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Karışıklık Matrisi")
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
plt.close()

# === ROC Eğrisi === #
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC eğrisi (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Şans çizgisi')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrisi')
plt.legend(loc='lower right')
plt.savefig(os.path.join(results_dir, "roc_curve.png"))
plt.close()

# === Precision-Recall Eğrisi === #
precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_score)
plt.figure()
plt.plot(recall_vals, precision_vals, label="Precision-Recall Eğrisi")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Eğrisi")
plt.legend()
plt.savefig(os.path.join(results_dir, "precision_recall_curve.png"))
plt.close()

print(f"📈 Eğitim ve test grafikleri kaydedildi: {results_dir}")
print(f"✅ Eğitim tamamlandı. Model ve metrikler kaydedildi.")