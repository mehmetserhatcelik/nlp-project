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
INPUT_FILE = "data/yargitay_dataset.json"         # Scraper'dan çıkan dosya
FAISS_INDEX_FILE = "data/index/yargitay_legal.index"     # Dense Index
BM25_FILE = "data/index/yargitay_bm25.pkl"               # Sparse Index
MAPPING_FILE = "data/yargitay_id_mapping.json"     # ID Eşleşmesi
MODEL_NAME = "KocLab-Bilkent/BERTurk-Legal"

# Cihaz ayarı (GPU varsa hızlanır)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"İşlem cihazı: {device}")

def clean_text(text):
    """BM25 için basit temizlik ve tokenization"""
    # Noktalama işaretlerini kaldır, küçült
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text.split()

def load_data():
    if not os.path.exists(INPUT_FILE):
        print(f"HATA: {INPUT_FILE} bulunamadı! Önce scraper kodunu çalıştırın.")
        return None
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def build_indices():
    data = load_data()
    if not data: return

    print(f"Toplam {len(data)} adet karar parçası (Chunk) indekslenecek.")

    # --- 1. AŞAMA: BM25 (SPARSE) INDEKSLEME ---
    print("\n[1/3] BM25 İndeksi Hazırlanıyor (Table II: Hybrid Retrieval)...")
    
    corpus_tokens = []
    ids = []
    
    for chunk in data:
        # STRATEJİ: Sybingco et al. [9] - Hibrit Arama için zenginleştirilmiş metin
        # Poly-Vector: "Yargıtay 3. HD 2021/100" ifadesini de indeksle ki numarayla arayan bulsun.
        citation = chunk['metadata']['poly_vector']['citation_label']
        
        # SAC: Özeti de ekle ki kelime bazlı arama (Örn: "tahliye") metinde geçmese bile özetten yakalasın.
        summary = chunk['metadata']['sac_context']['parent_summary']
        
        # Ana Metin
        text = chunk['text']
        
        # Hepsini Birleştir
        combo_text = f"{citation} {summary} {text}"
        
        corpus_tokens.append(clean_text(combo_text))
        ids.append(chunk["id"])
        
    bm25 = BM25Okapi(corpus_tokens)
    
    with open(BM25_FILE, "wb") as f:
        pickle.dump({"model": bm25, "ids": ids}, f)
    print(f"   -> BM25 kaydedildi: {BM25_FILE}")

    # --- 2. AŞAMA: DENSE (FAISS) INDEKSLEME ---
    print("\n[2/3] Dense (Vektör) İndeksi Hazırlanıyor (Table I: SAC & Poly-Vector)...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME).to(device)
        model.eval()
    except Exception as e:
        print(f"Model yüklenemedi: {e}")
        return

    embeddings = []
    batch_size = 16 # Bellek durumuna göre artırılabilir (32, 64)
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batch_texts = []
        
        for d in batch:
            # STRATEJİ: Reuter et al. [2] - Summary Augmented Chunking (SAC)
            # Embedding = Vector(Citation + Summary + Text)
            # Bu sayede vektör uzayında "parça" hem kimliğini (Citation) hem bağlamını (Summary) bilir.
            
            cit = d['metadata']['poly_vector']['citation_label']
            summ = d['metadata']['sac_context']['parent_summary']
            txt = d['text']
            
            # Modelin anlayacağı format
            input_txt = f"{cit}. {summ} {txt}"
            batch_texts.append(input_txt)
            
        # Tokenize & GPU'ya at
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Mean Pooling (Hukuk metinleri için CLS'den daha stabil)
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            batch_vecs = sum_embeddings / sum_mask
            
            # CPU'ya geri al ve listeye ekle
            embeddings.append(batch_vecs.cpu().numpy())
            
        if i % 100 == 0 and i > 0:
            print(f"   -> {i} parça işlendi...")

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
    
    faiss.write_index(index, FAISS_INDEX_FILE)
    
    # ID Mapping'i kaydet (FAISS sadece int ID tutar, biz string ID'leri burada saklıyoruz)
    with open(MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(ids, f)
        
    print(f"   -> FAISS İndeksi Hazır: {FAISS_INDEX_FILE}")
    print(f"   -> ID Eşleşmesi Hazır: {MAPPING_FILE}")
    print("\n✅ TÜM İŞLEMLER TAMAMLANDI!")

if __name__ == "__main__":
    build_indices()