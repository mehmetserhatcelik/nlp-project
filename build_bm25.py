import json
import pickle
from rank_bm25 import BM25Okapi
import re
import glob
from pathlib import Path

# --- AYARLAR ---
BM25_FILE = "data/index/tbk_bm25.pkl"

# Target laws to process (matching build_dataset.py)
TARGET_LAWS = [
    "4721",  # TMK - Türk Medeni Kanunu
    "5237",  # TCK - Türk Ceza Kanunu
    "6102",  # TTK - Türk Ticaret Kanunu
    "4857",  # İş Kanunuæ
    "6098",  # TBK - Türk Borçlar Kanunu
]

def clean_text(text):
    # Basit bir Türkçe tokenizer simülasyonu
    text = text.lower()
    # Noktalama işaretlerini kaldır
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

def load_all_datasets():
    """
    Load all law datasets and merge them into a single list.
    Returns merged data list.
    """
    data_dir = Path("data")
    all_data = []
    
    print("Tüm kanun datasetleri yükleniyor...")
    
    for law_code in TARGET_LAWS:
        dataset_file = data_dir / f"{law_code}_dataset.json"
        
        if dataset_file.exists():
            print(f"  -> {dataset_file.name} yükleniyor...")
            with open(dataset_file, "r", encoding="utf-8") as f:
                law_data = json.load(f)
                all_data.extend(law_data)
                print(f"     {len(law_data)} chunk eklendi")
        else:
            print(f"  ⚠️  {dataset_file.name} bulunamadı, atlanıyor...")
    
    print(f"\nToplam {len(all_data)} chunk yüklendi (tüm kanunlar birleştirildi)")
    return all_data

def build_bm25():
    print("=" * 60)
    print("BM25 İNDEKS OLUŞTURMA (TÜM KANUNLAR)")
    print("=" * 60)
    
    # Load all datasets
    data = load_all_datasets()
    
    if not data:
        print("HATA: Hiç veri yüklenemedi! Lütfen önce build_dataset.py'yi çalıştırın.")
        return
    
    corpus = []
    ids = []
    
    print("\nBM25 için corpus hazırlanıyor...")
    for chunk in data:
        # Metin + Citation Label + Özet birleşimi
        # Literatürdeki Sybingco et al. yaklaşımı: Hem metni hem metadata'yı indeksle
        full_text = f"{chunk['metadata']['poly_vector']['citation_label']} {chunk['text']} {chunk['metadata']['sac_context']['parent_summary']}"
        
        tokenized_doc = clean_text(full_text)
        corpus.append(tokenized_doc)
        ids.append(chunk["id"])

    print(f"\nBM25 modeli eğitiliyor ({len(corpus)} döküman)...")
    bm25 = BM25Okapi(corpus)
    
    # Modeli ve ID listesini kaydet
    with open(BM25_FILE, "wb") as f:
        pickle.dump({"model": bm25, "ids": ids}, f)
        
    print(f"✅ BM25 indeksi '{BM25_FILE}' olarak kaydedildi.")
    print(f"   Toplam {len(ids)} doküman indekslendi")

if __name__ == "__main__":
    build_bm25()