import glob

transcripts = []

text_files = glob.glob("C:/Users/HAZEL/LibriSpeech/LibriSpeech/train-clean-100/*/*/*.txt") + \
             glob.glob("C:/Users/HAZEL/LibriSpeech/LibriSpeech/train-clean-360/*/*/*.txt")

for txt_file in text_files:
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                transcripts.append(parts[1])
