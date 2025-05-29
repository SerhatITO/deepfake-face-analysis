import os

# mfcc_cache klasörünün yolu
mfcc_dir = r"C:\Users\HAZEL\Desktop\mfcc_cache"

# Etiket sayılarını saymak için sayaçlar
label_counts = {}

# Dosyaları dolaş
for filename in os.listdir(mfcc_dir):
    if filename.endswith(".npy"):
        # Örnek: dosya adı içinde "fake" veya "real" geçiyorsa
        if "fake" in filename:
            label = 0
        elif "real" in filename:
            label = 1
        else:
            continue  # sınıf etiketi yoksa atla

        # Sayaç güncelle
        label_counts[label] = label_counts.get(label, 0) + 1

# Sonuçları yazdır
for label, count in label_counts.items():
    print(f"Label {label}: {count} örnek")

toplam = sum(label_counts.values())
print(f"Toplam: {toplam} örnek")
