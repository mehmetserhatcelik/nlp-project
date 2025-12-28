import json
import pickle
from rank_bm25 import BM25Okapi
import re

# --- AYARLAR ---
INPUT_FILE = "data/tbk_dataset.json"
BM25_FILE = "data/index/tbk_bm25.pkl"

def clean_text(text):
    # Basit bir Türkçe tokenizer simülasyonu
    text = text.lower()
    # Noktalama işaretlerini kaldır
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

def build_bm25():
    print("Veri yükleniyor...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    corpus = []
    ids = []
    
    print("BM25 için corpus hazırlanıyor...")
    for chunk in data:
        # Metin + Citation Label + Özet birleşimi
        # Literatürdeki Sybingco et al. yaklaşımı: Hem metni hem metadata'yı indeksle
        full_text = f"{chunk['metadata']['poly_vector']['citation_label']} {chunk['text']} {chunk['metadata']['sac_context']['parent_summary']}"
        
        tokenized_doc = clean_text(full_text)
        corpus.append(tokenized_doc)
        ids.append(chunk["id"])

    print(f"BM25 modeli eğitiliyor ({len(corpus)} döküman)...")
    bm25 = BM25Okapi(corpus)
    
    # Modeli ve ID listesini kaydet
    with open(BM25_FILE, "wb") as f:
        pickle.dump({"model": bm25, "ids": ids}, f)
        
    print(f"BM25 indeksi '{BM25_FILE}' olarak kaydedildi.")

if __name__ == "__main__":
    build_bm25()