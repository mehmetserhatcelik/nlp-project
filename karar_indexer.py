import json
import torch
import numpy as np
import faiss
import pickle
import os
import re
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi

# --- YAPILANDIRMA ---
# Yargıtay dataset
YARGITAY_INPUT_FILE = "data/yargitay_dataset.json"
YARGITAY_FAISS_INDEX_FILE = "data/index/yargitay_legal.index"
YARGITAY_BM25_FILE = "data/index/yargitay_bm25.pkl"
YARGITAY_MAPPING_FILE = "data/yargitay_id_mapping.json"

# Similar Cases dataset
SIMILAR_CASES_INPUT_FILE = "data/similar_cases.json"
SIMILAR_CASES_FAISS_INDEX_FILE = "data/index/similar_cases_legal.index"
SIMILAR_CASES_BM25_FILE = "data/index/similar_cases_bm25.pkl"
SIMILAR_CASES_MAPPING_FILE = "data/similar_cases_id_mapping.json"

MODEL_NAME = "KocLab-Bilkent/BERTurk-Legal"

# Cihaz ayarı (GPU varsa hızlanır)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"İşlem cihazı: {device}")

def clean_text(text):
    """BM25 için basit temizlik ve tokenization"""
    # Basit bir Türkçe tokenizer simülasyonu
    text = text.lower()
    # Noktalama işaretlerini kaldır
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

def load_data(input_file, dataset_name):
    """Dataset dosyasını yükle"""
    if not os.path.exists(input_file):
        print(f"HATA: {input_file} bulunamadı!")
        return None
    
    print(f"{dataset_name} dataset yükleniyor: {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  -> {len(data)} karar yüklendi")
    return data

def generate_id(decision, index, prefix="YARGITAY"):
    """Her karar için benzersiz bir ID oluştur"""
    # URL'den veya case/decision number'dan ID oluştur
    if decision.get('url'):
        # URL'den ID çıkar (örn: "GE95G9JAgTI" kısmı)
        url = decision['url']
        if 'v=' in url:
            url_id = url.split('v=')[-1]
            return f"{prefix}:{url_id}"
    
    # Alternatif: case_number ve decision_number kullan
    case_num = decision.get('case_number', '').strip().replace(' ', '_')
    decision_num = decision.get('decision_number', '').strip().replace(' ', '_')
    chamber = decision.get('chamber', '').strip().replace(' ', '_').replace('.', '')
    
    if case_num and decision_num:
        return f"{prefix}:{chamber}_{case_num}_{decision_num}"
    
    # Son çare: index kullan
    return f"{prefix}:{index}"

def build_indices_for_dataset(input_file, faiss_index_file, bm25_file, mapping_file, dataset_name, id_prefix):
    """Belirtilen dataset için BM25 ve FAISS indexleri oluştur"""
    print("=" * 60)
    print(f"{dataset_name.upper()} KARAR İNDEKS OLUŞTURMA")
    print("=" * 60)
    
    data = load_data(input_file, dataset_name)
    if not data:
        print(f"HATA: {dataset_name} verisi yüklenemedi!")
        return False

    print(f"\nToplam {len(data)} adet karar indekslenecek.")

    # --- 1. AŞAMA: BM25 (SPARSE) INDEKSLEME ---
    print("\n[1/3] BM25 İndeksi Hazırlanıyor...")
    
    corpus_tokens = []
    ids = []
    
    print("BM25 için corpus hazırlanıyor...")
    for i, decision in enumerate(data):
        # Full text üzerinden indexleme
        full_text = decision.get('full_text', '')
        
        # Eğer full_text yoksa, diğer alanları birleştir
        if not full_text:
            summary = decision.get('summary', '') or ''
            conclusion = decision.get('conclusion', '') or ''
            full_text = f"{summary} {conclusion}".strip()
        
        tokenized_doc = clean_text(full_text)
        corpus_tokens.append(tokenized_doc)
        
        # ID oluştur
        decision_id = generate_id(decision, i, prefix=id_prefix)
        ids.append(decision_id)
        
    print(f"\nBM25 modeli eğitiliyor ({len(corpus_tokens)} doküman)...")
    bm25 = BM25Okapi(corpus_tokens)
    
    # Modeli ve ID listesini kaydet
    os.makedirs(os.path.dirname(bm25_file), exist_ok=True)
    with open(bm25_file, "wb") as f:
        pickle.dump({"model": bm25, "ids": ids}, f)
    print(f"✅ BM25 indeksi '{bm25_file}' olarak kaydedildi.")
    print(f"   Toplam {len(ids)} doküman indekslendi")

    # --- 2. AŞAMA: DENSE (FAISS) INDEKSLEME ---
    print("\n[2/3] Dense (Vektör) İndeksi Hazırlanıyor (Full Text Embedding)...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME).to(device)
        model.eval()
        print(f"Model yüklendi: {MODEL_NAME}")
    except Exception as e:
        print(f"Model yüklenemedi: {e}")
        return False

    embeddings = []
    ids = []  # ID listesi
    batch_size = 16  # Bellek durumuna göre artırılabilir (32, 64)
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batch_texts = []
        
        for j, decision in enumerate(batch):
            # Full text üzerinden indexleme
            full_text = decision.get('full_text', '')
            
            # Eğer full_text yoksa, diğer alanları birleştir
            if not full_text:
                summary = decision.get('summary', '') or ''
                conclusion = decision.get('conclusion', '') or ''
                full_text = f"{summary} {conclusion}".strip()
            
            batch_texts.append(full_text)
            
            # ID oluştur
            decision_id = generate_id(decision, i + j, prefix=id_prefix)
            ids.append(decision_id)
        
        # Vektörleri işle
        if batch_texts:
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            batch_embeddings = sum_embeddings / sum_mask
            embeddings.append(batch_embeddings.cpu().numpy())
            
        if i % (batch_size * 10) == 0 and i > 0:
            print(f"   -> {i} karar işlendi...")

    # --- 3. AŞAMA: FAISS KAYIT ---
    print("\n[3/3] FAISS İndeksi Diske Yazılıyor...")
    
    # Tüm batchleri birleştir
    all_vecs = np.vstack(embeddings)
    
    # Cosine Similarity için L2 Normalizasyonu şart
    faiss.normalize_L2(all_vecs)
    
    # Boyut (768 for BERT-Base)
    d = all_vecs.shape[1]
    
    # IndexFlatIP = Inner Product (Normalize vektörlerde Cosine Similarity'e eşittir)
    index = faiss.IndexFlatIP(d)
    index.add(all_vecs)
    
    os.makedirs(os.path.dirname(faiss_index_file), exist_ok=True)
    faiss.write_index(index, faiss_index_file)
    
    # ID Mapping'i kaydet
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(ids, f)
        
    print(f"✅ FAISS İndeksi Hazır: {faiss_index_file} (Toplam {len(ids)} vektör)")
    print(f"✅ ID Eşleşmesi Hazır: {mapping_file}")
    print(f"\n✅ {dataset_name.upper()} İŞLEMLERİ TAMAMLANDI!")
    return True

def build_indices():
    """Sadece Similar Cases için indexleri oluştur"""
    # Similar Cases dataset için index oluştur
    success = build_indices_for_dataset(
        SIMILAR_CASES_INPUT_FILE,
        SIMILAR_CASES_FAISS_INDEX_FILE,
        SIMILAR_CASES_BM25_FILE,
        SIMILAR_CASES_MAPPING_FILE,
        "Similar Cases",
        "SIMILAR_CASES"
    )
    
    if success:
        print("\n✅ SIMILAR CASES İŞLEMLERİ BAŞARIYLA TAMAMLANDI!")
    else:
        print("\n⚠️  SIMILAR CASES İŞLEMLERİ TAMAMLANAMADI!")

if __name__ == "__main__":
    build_indices()