import json
import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel

# --- AYARLAR ---
FAISS_INDEX_FILE = "data/index/tbk_legal.index"
MAPPING_FILE = "data/id_mapping.json"

# Target laws to process (matching build_dataset.py)
TARGET_LAWS = [
    "4721",  # TMK - Türk Medeni Kanunu
    "5237",  # TCK - Türk Ceza Kanunu
    "6102",  # TTK - Türk Ticaret Kanunu
    "4857",  # İş Kanunu
    "6098",  # TBK - Türk Borçlar Kanunu
]

# PROJE SEÇİMİ: BERTurk-Legal
# KocLab-Bilkent tarafından Türk hukuku metinleri için özelleştirilmiş BERT modeli
# Zero-shot classification ile Yargıtay kararları üzerinde state-of-the-art sonuçlar veriyor
MODEL_NAME = "KocLab-Bilkent/BERTurk-Legal"

# Cihaz Seçimi (GPU varsa kullan, yoksa CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"İşlem yapılacak cihaz: {device}")

def load_data():
    """
    Load all law datasets and merge them into a single list.
    Returns merged data list.
    """
    from pathlib import Path
    
    data_dir = Path("data")
    all_data = []
    
    print("Tüm kanun datasetleri yükleniyor...")
    
    for law_code in TARGET_LAWS:
        dataset_file = data_dir / f"{law_code}_dataset.json"
        
        if dataset_file.exists():
            print(f"  -> {dataset_file.name} yükleniyor...")
            try:
                with open(dataset_file, "r", encoding="utf-8") as f:
                    law_data = json.load(f)
                    all_data.extend(law_data)
                    print(f"     {len(law_data)} chunk eklendi")
            except Exception as e:
                print(f"     ⚠️  Hata: {e}")
        else:
            print(f"  ⚠️  {dataset_file.name} bulunamadı, atlanıyor...")
    
    if not all_data:
        print(f"\nHATA: Hiç veri yüklenemedi! Lütfen önce build_dataset.py'yi çalıştırın.")
        exit(1)
    
    print(f"\nToplam {len(all_data)} adet veri yüklendi (tüm kanunlar birleştirildi).")
    return all_data

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
    """
    True Poly-Vector Indexing: Her chunk için iki ayrı vektör oluşturur.
    1. Content Vector: summary + text (anlamsal içerik)
    2. Citation Vector: citation_label (tam atıf referansı, örn: "TBK 6098 Madde 12")
    """
    embeddings = []
    ids = []
    
    print("Poly-Vector Vektörleştirme başlıyor (Her chunk için 2 vektör)...")
    model.eval() # Modeli inference moduna al
    
    # Batch processing (Hız için veriyi gruplar halinde işle)
    BATCH_SIZE = 16 
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        batch_content_texts = []  # Content vektörleri için
        batch_citation_texts = []  # Citation vektörleri için
        
        for chunk in batch:
            summary = chunk["metadata"]["sac_context"]["parent_summary"]
            text = chunk["text"]
            citation = chunk["metadata"]["poly_vector"]["citation_label"]
            
            # 1. Content Vector: Özet + Metin (anlamsal içerik için)
            content_text = f"{summary} {text}"
            batch_content_texts.append(content_text)
            
            # 2. Citation Vector: Sadece atıf etiketi (tam referans için, örn: "TBK 6098 Madde 12")
            citation_text = f"{citation}"
            batch_citation_texts.append(citation_text)
            
            # Her chunk için aynı ID'yi iki kez ekle (content ve citation için)
            ids.append(chunk["id"])
            ids.append(chunk["id"])

        # Content vektörlerini işle
        if batch_content_texts:
            inputs_content = tokenizer(
                batch_content_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(device)
            
            with torch.no_grad():
                outputs_content = model(**inputs_content)
            
            attention_mask_content = inputs_content['attention_mask']
            token_embeddings_content = outputs_content.last_hidden_state
            input_mask_expanded_content = attention_mask_content.unsqueeze(-1).expand(token_embeddings_content.size()).float()
            sum_embeddings_content = torch.sum(token_embeddings_content * input_mask_expanded_content, 1)
            sum_mask_content = torch.clamp(input_mask_expanded_content.sum(1), min=1e-9)
            batch_embeddings_content = sum_embeddings_content / sum_mask_content
            embeddings.append(batch_embeddings_content.cpu().numpy())
        
        # Citation vektörlerini işle
        if batch_citation_texts:
            inputs_citation = tokenizer(
                batch_citation_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(device)
            
            with torch.no_grad():
                outputs_citation = model(**inputs_citation)
            
            attention_mask_citation = inputs_citation['attention_mask']
            token_embeddings_citation = outputs_citation.last_hidden_state
            input_mask_expanded_citation = attention_mask_citation.unsqueeze(-1).expand(token_embeddings_citation.size()).float()
            sum_embeddings_citation = torch.sum(token_embeddings_citation * input_mask_expanded_citation, 1)
            sum_mask_citation = torch.clamp(input_mask_expanded_citation.sum(1), min=1e-9)
            batch_embeddings_citation = sum_embeddings_citation / sum_mask_citation
            embeddings.append(batch_embeddings_citation.cpu().numpy())
        
        if i % 100 == 0 and i > 0:
            print(f"{i} parça işlendi ({i*2} vektör oluşturuldu)...")

    # Tüm batch'leri tek bir numpy array'de birleştir
    all_embeddings = np.vstack(embeddings)
    print(f"Toplam {len(ids)} vektör oluşturuldu ({len(chunks)} chunk x 2 vektör)")
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