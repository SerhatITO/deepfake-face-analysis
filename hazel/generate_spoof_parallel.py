import os
from TTS.api import TTS
import concurrent.futures
from transcripts import transcripts  # transcripts listesi burada

tts = TTS("tts_models/multilingual/multi-dataset/your_tts")

batch_size = 1000     # Her batch'te kaç cümle işlenecek
num_workers = 8       # Paralel çalışan thread sayısı

output_dir = "spoof_audio"
os.makedirs(output_dir, exist_ok=True)

def generate_tts(index_sentence):
    index, sentence = index_sentence
    file_path = os.path.join(output_dir, f"spoof_{index}.wav")
    try:
        tts.tts_to_file(text=sentence, file_path=file_path, speaker="female-en-5", language="en")
        return f"Saved: {file_path}"
    except Exception as e:
        return f"Error at index {index}: {str(e)}"

def batch_done_flag_path(batch_num):
    # Batch tamamlandıktan sonra oluşturulacak "flag" dosyasının yolu
    return os.path.join(output_dir, f"batch_done_{batch_num}.flag")

def main():
    sentences = transcripts[:60000]
    total_batches = (len(sentences) + batch_size - 1) // batch_size

    print(f"Toplam cümle sayısı: {len(sentences)}")
    print(f"Toplam batch sayısı: {total_batches}")

    for batch_num in range(total_batches):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, len(sentences))
        batch = list(enumerate(sentences[batch_start:batch_end], start=batch_start))

        flag_path = batch_done_flag_path(batch_num)
        if os.path.exists(flag_path):
            print(f"Batch {batch_num + 1} daha önce tamamlanmış, atlanıyor (flag dosyası bulundu).")
            continue  # Batch tamamen atlanıyor

        print(f"Batch {batch_num + 1} işleniyor (cümleler {batch_start} - {batch_end - 1})")

        # Eksik olan wav dosyalarını bul, sadece eksik olanlar işlenecek
        missing_items = [(i, s) for i, s in batch if not os.path.exists(os.path.join(output_dir, f"spoof_{i}.wav"))]

        if not missing_items:
            # Eğer eksik yoksa flag dosyasını oluştur, batch tamam demek
            open(flag_path, 'w').close()
            print(f"Batch {batch_num + 1} tamamen zaten var, flag dosyası oluşturuldu.")
            continue

        # Paralel olarak eksik dosyaları üret
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(generate_tts, missing_items))

        # Sonuçları yazdır
        for res in results:
            print(res)

        # Batch tamamlandıktan sonra flag dosyasını oluştur
        open(flag_path, 'w').close()
        print(f"Batch {batch_num + 1} tamamlandı, flag dosyası oluşturuldu.")

if __name__ == "__main__":
    main()
