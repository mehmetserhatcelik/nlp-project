import json
import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel

# --- AYARLAR ---
INPUT_FILE = "data/tbk_dataset.json"
FAISS_INDEX_FILE = "data/index/tbk_legal.index"
MAPPING_FILE = "data/id_mapping.json"

# PROJE SEÇİMİ: BERTurk-Legal
# KocLab-Bilkent tarafından Türk hukuku metinleri için özelleştirilmiş BERT modeli
# Zero-shot classification ile Yargıtay kararları üzerinde state-of-the-art sonuçlar veriyor
MODEL_NAME = "KocLab-Bilkent/BERTurk-Legal"

# Cihaz Seçimi (GPU varsa kullan, yoksa CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"İşlem yapılacak cihaz: {device}")

def load_data():
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"{len(data)} adet veri yüklendi.")
        return data
    except FileNotFoundError:
        print(f"HATA: {INPUT_FILE} bulunamadı. Lütfen önce dataset oluşturma kodunu çalıştırın.")
        exit()

def get_embedding_model():
    print(f"Model yükleniyor: {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME).to(device)
        return tokenizer, model
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        print("Model ismini kontrol edin veya 'dbmdz/bert-base-turkish-cased' kullanmayı deneyin.")
        exit()

def generate_embeddings(chunks, tokenizer, model):
    embeddings = []
    ids = []
    
    print("Vektörleştirme başlıyor...")
    model.eval() # Modeli inference moduna al
    
    # Batch processing (Hız için veriyi gruplar halinde işle)
    BATCH_SIZE = 16 
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        batch_texts = []
        
        for chunk in batch:
            # --- SAC (Summary-Augmented Chunking) Entegrasyonu  ---
            # Modelin "bağlamı" anlaması için Özet + Atıf Etiketi + Metin birleştiriliyor.
            summary = chunk["metadata"]["sac_context"]["parent_summary"]
            text = chunk["text"]
            citation = chunk["metadata"]["poly_vector"]["citation_label"] # 
            
            # Girdi formatı: "TBK 6098 Madde 12. Özet... Metin..."
            combined_text = f"{citation}. {summary} {text}"
            batch_texts.append(combined_text)
            
            # ID'yi sakla
            ids.append(chunk["id"])

        # Tokenize
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Mean Pooling: Kelime vektörlerinin ortalamasını alarak cümle vektörü oluşturma
        # (Legal metinlerde CLS token yerine Mean Pooling genelde daha iyi sonuç verir)
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        
        # Maskelenmiş (padding olan) kısımları hesaba katmadan ortalama al
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        batch_embeddings = sum_embeddings / sum_mask
        
        # CPU'ya geri al ve listeye ekle
        embeddings.append(batch_embeddings.cpu().numpy())
        
        if i % 100 == 0 and i > 0:
            print(f"{i} parça işlendi...")

    # Tüm batch'leri tek bir numpy array'de birleştir
    all_embeddings = np.vstack(embeddings)
    return all_embeddings, ids

def save_faiss_index(embeddings, ids):
    print("FAISS İndeksi oluşturuluyor...")
    
    # Embedding boyutu (BERT base için genelde 768)
    d = embeddings.shape[1]
    
    # Normalizasyon (Cosine Similarity için gerekli)
    faiss.normalize_L2(embeddings)
    
    # IndexFlatIP = Inner Product (İç Çarpım) -> Normalize edilmiş veriyle Cosine Similarity olur
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    
    print(f"Toplam {index.ntotal} vektör indekslendi.")
    
    # İndeksi kaydet
    faiss.write_index(index, FAISS_INDEX_FILE)
    
    # Mapping dosyasını kaydet (Hangi ID hangi vektöre denk geliyor)
    with open(MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(ids, f)
        
    print(f"Tamamlandı! Dosyalar: {FAISS_INDEX_FILE}, {MAPPING_FILE}")

if __name__ == "__main__":
    # 1. Veriyi Yükle
    data = load_data()
    
    # 2. Modeli Hazırla
    tokenizer, model = get_embedding_model()
    
    # 3. Embedding Oluştur
    vectors, id_list = generate_embeddings(data, tokenizer, model)
    
    # 4. FAISS'e Kaydet
    save_faiss_index(vectors, id_list)