# Deepfake Face Analysis

Bu proje, deepfake tespiti için videolardan yüz çıkarımı, kırpma ve sınıflandırma süreçlerini içerir. Projenin amacı, sahte ve gerçek videoları ayrıştırabilen bir model geliştirmektir.

---

## 🔍 Proje Adımları

### 1️⃣ Ön İşleme
- `scripts/extract_frames.py`  
  → Videolardan kare (frame) çıkarır  
- `scripts/crop_faces.py`  
  → Kareler üzerinden yüzleri tespit eder ve kırpar  
- Kırpılan yüzler `faces/real` ve `faces/fake` klasörlerine kaydedilir

### 2️⃣ Model Eğitimi
- `scripts/train.py`  
  → Yüz görüntülerini kullanarak derin öğrenme modeli eğitir  
  (Model: ResNet18 - Transfer Learning)

---

## 🧠 Kullanılan Veri Setleri

- **Celeb-DF-v2**  
- **FaceForensics++ (c40 sıkıştırılmış videolar)**  

Veriler `data/` klasöründe tutulmakta; yüzler `faces/` altında `real` ve `fake` olarak ayrılmıştır.

---

## 💻 Gereksinimler

Tüm kütüphaneler `scripts/requirements.txt` dosyasında listelenmiştir. Ortamı kurmak için:

```bash
pip install -r scripts/requirements.txt

## 🧠 Model Eğitimi

Model olarak ResNet18 kullanılmıştır. Eğitim verileri `faces/real` ve `faces/fake` klasörlerinden alınarak `scripts/train_model.py` dosyasıyla eğitilir.

### Eğitim için:
```bash
python scripts/train_model.py

