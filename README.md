# Deepfake Face Analysis

Bu proje, videolardaki yüzleri karelere ayırarak (frame extraction) ve yüzleri kırparak (face crop) deepfake tespiti için ön işleme yapılmasını sağlar.

## 🔍 Proje İçeriği

- `scripts/extract_frames.py` → Videolardan kare çıkarır (frame extraction)
- `scripts/crop_faces.py` → Her kareden yüz tespiti ve kırpma işlemi yapar
- `test_main.py` → Örnek bir videodan frame çıkarımı yapar
- `test_crop.py` → Çıkarılan karelerden yüzleri kırpar

## 🧠 Kullanılan Veri Seti

Veri seti: **Celeb-DF v2**  
İçeriği şu şekildedir:

