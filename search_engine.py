import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import pickle
import json
import numpy as np
import re
import os
from dotenv import load_dotenv
from openai import OpenAI

# .env dosyasını yükle
load_dotenv()

# --- DOSYA AYARLARI ---
# 1. KANUN (Statute) Dosyaları
STATUTE_FAISS   = "data/index/tbk_legal.index"
STATUTE_BM25    = "data/index/tbk_bm25.pkl"
STATUTE_MAPPING = "data/id_mapping.json"
STATUTE_DATA    = "data/tbk_dataset.json"

# 2. EMSAL KARAR (Case) Dosyaları (Yargıtay Scraper ve Indexer çıktıları)
CASE_FAISS      = "data/index/yargitay_legal.index"
CASE_BM25       = "data/index/yargitay_bm25.pkl"
CASE_MAPPING    = "data/yargitay_id_mapping.json"
CASE_DATA       = "data/yargitay_dataset.json"

MODEL_NAME = "KocLab-Bilkent/BERTurk-Legal"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY .env dosyasında bulunamadı!")
class StatuteManager:
    """
    Bu sınıfın görevi: Yargıtay kararında geçen "LAW:6098-12" gibi bir atfı alıp,
    TBK veri setimizdeki parça parça (fıkra) metinleri birleştirerek tam maddeyi getirmektir.
    """
    def __init__(self, data_path):
        print("   -> Kanun Metinleri Yükleniyor (StatuteManager)...")
        self.articles = {} 
        self._load_data(data_path)

    def _load_data(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for chunk in data:
            # Chunk ID: "LAW:6098-12-1" -> Biz bunu "LAW:6098-12" (Madde) çatısı altında topluyoruz.
            parts = chunk['id'].split('-')
            if len(parts) >= 3: # LAW - NO - MADDE - FIKRA
                parent_id = f"{parts[0]}-{parts[1]}" # Örn: LAW:6098-12
                
                if parent_id not in self.articles:
                    self.articles[parent_id] = []
                
                # Fıkra sırası için numarayı al
                para_no = int(parts[2]) if parts[2].isdigit() else 0
                self.articles[parent_id].append((para_no, chunk['text']))

    def get_law_text(self, law_id):
        """Verilen ID (LAW:6098-12) için birleştirilmiş metni döner."""
        # Eğer direkt fıkra ID'si geldiyse (LAW:6098-12-1), madde ID'sine çevir
        if law_id.count('-') > 1:
            law_id = "-".join(law_id.split('-')[:2])

        if law_id in self.articles:
            # Fıkraları sıraya diz
            sorted_paras = sorted(self.articles[law_id], key=lambda x: x[0])
            full_text = ""
            for num, text in sorted_paras:
                prefix = f"({num}) " if num > 0 else ""
                full_text += f"{prefix}{text}\n"
            return full_text.strip()
        return None

class HybridLegalAssistant:
    def __init__(self):
        print("\n🚀 SİSTEM BAŞLATILIYOR...")
        self.device = torch.device("cpu")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # 1. Modeli Yükle
        print(f"   -> Model Yükleniyor: {MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)

        # 2. Kanun İndekslerini Yükle
        print("   -> Kanun İndeksleri Yükleniyor...")
        self.statute_index = faiss.read_index(STATUTE_FAISS)
        self.statute_ids = json.load(open(STATUTE_MAPPING, "r", encoding="utf-8"))
        self.statute_bm25_data = pickle.load(open(STATUTE_BM25, "rb"))
        self.statute_data = {d['id']: d for d in json.load(open(STATUTE_DATA, "r", encoding="utf-8"))}
        
        # 3. Emsal Karar İndekslerini Yükle
        print("   -> Emsal Karar İndeksleri Yükleniyor...")
        try:
            self.case_index = faiss.read_index(CASE_FAISS)
            self.case_ids = json.load(open(CASE_MAPPING, "r", encoding="utf-8"))
            self.case_bm25_data = pickle.load(open(CASE_BM25, "rb"))
            self.case_data = {d['id']: d for d in json.load(open(CASE_DATA, "r", encoding="utf-8"))}
        except Exception as e:
            print(f"UYARI: Emsal karar dosyaları bulunamadı ({e}). Sadece kanun çalışacak.")
            self.case_index = None

        # 4. Statute Manager (Bağlantı Kurucu)
        self.statute_manager = StatuteManager(STATUTE_DATA)
        print("✅ Sistem Hazır!\n")

    def _generate_legal_query(self, user_text):
        """Kullanıcı hikayesini hukuki terimlere çevirir."""
        system_prompt = """Sen uzman bir Türk Hukukçusun. Kullanıcının anlattığı olayı, hem Kanun Maddesi (Mevzuat) hem de Yargıtay Emsal Kararı aramak için en uygun anahtar kelimelere ve hukuki terimlere dönüştür. Sadece sorguyu yaz."""
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system_prompt}, 
                          {"role": "user", "content": user_text}],
                temperature=0.3
            )
            return resp.choices[0].message.content.strip()
        except: return user_text

    def _get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean Pooling
            attn_mask = inputs['attention_mask']
            token_embeds = outputs.last_hidden_state
            mask_expanded = attn_mask.unsqueeze(-1).expand(token_embeds.size()).float()
            vec = torch.sum(token_embeds * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
            return vec.cpu().numpy()

    def _search_generic(self, query, faiss_index, id_map, bm25_model, bm25_ids, k=3, alpha=0.7):
        """Hem Kanun hem Karar için ortak hibrit arama fonksiyonu"""
        # Dense Search
        q_vec = self._get_embedding(query)
        faiss.normalize_L2(q_vec)
        dists, idxs = faiss_index.search(q_vec, k*2)
        dense_res = {id_map[i]: float(dists[0][j]) for j, i in enumerate(idxs[0]) if i != -1}

        # Sparse Search
        tokenized_q = query.lower().split()
        scores = bm25_model.get_scores(tokenized_q)
        top_n = np.argsort(scores)[::-1][:k*2]
        sparse_res = {bm25_ids[i]: float(scores[i]) for i in top_n}

        # Fusion
        results = []
        all_ids = set(dense_res.keys()) | set(sparse_res.keys())
        max_bm25 = max(sparse_res.values()) if sparse_res else 1.0

        for doc_id in all_ids:
            s_dense = dense_res.get(doc_id, 0.0)
            s_sparse = sparse_res.get(doc_id, 0.0) / max_bm25 if max_bm25 > 0 else 0
            score = (alpha * s_dense) + ((1 - alpha) * s_sparse)
            results.append((doc_id, score))
        
        return sorted(results, key=lambda x: x[1], reverse=True)[:k]

    def solve_case(self, user_case):
        print("="*60)
        print(f"📝 OLAY: {user_case[:100]}...")
        
        # 1. Query Generation
        legal_query = self._generate_legal_query(user_case)
        print(f"🔍 HUKUKİ SORGU: {legal_query}")
        print("="*60)

        # --- BÖLÜM 1: İLGİLİ KANUN MADDELERİ ---
        print("\n" + "⚖️  İLGİLİ KANUN MADDELERİ (MEVZUAT)".center(60, "-"))
        statute_results = self._search_generic(
            legal_query, self.statute_index, self.statute_ids, 
            self.statute_bm25_data["model"], self.statute_bm25_data["ids"], k=3
        )

        for rank, (doc_id, score) in enumerate(statute_results):
            data = self.statute_data[doc_id]
            print(f"\n#{rank+1} [{data['metadata']['poly_vector']['citation_label']}] (Skor: {score:.4f})")
            print(f"   ÖZET: {data['metadata']['sac_context']['parent_summary']}")
            print(f"   METİN: {data['text'][:150]}...")

        # --- BÖLÜM 2: EMSAL KARARLAR ---
        if self.case_index:
            print("\n" + "🏛️  EMSAL YARGITAY KARARLARI".center(60, "-"))
            case_results = self._search_generic(
                legal_query, self.case_index, self.case_ids,
                self.case_bm25_data["model"], self.case_bm25_data["ids"], k=3
            )

            for rank, (doc_id, score) in enumerate(case_results):
                data = self.case_data[doc_id]
                cit = data['metadata']['poly_vector']['citation_label']
                print(f"\n#{rank+1} [{cit}] (Skor: {score:.4f})")
                print(f"   KARAR ÖZETİ: {data['metadata']['sac_context']['parent_summary']}")
                print(f"   GEREKÇE: {data['text'][:200]}...")
                
                # --- KRİTİK BÖLÜM: KARARIN İÇİNDEKİ KANUNLARI BUL VE GÖSTER ---
                links = data['metadata'].get('graph_links', [])
                if links:
                    print(f"   🔗 BU KARARDA UYGULANAN KANUNLAR:")
                    seen_laws = set()
                    for link in links:
                        target_id = link['target_id'] # Örn: LAW:6098-12
                        
                        # Aynı maddeyi tekrar tekrar basma
                        if target_id in seen_laws: continue
                        seen_laws.add(target_id)

                        # 1. Kanun bizim veritabanımızda var mı? (StatuteManager ile kontrol et)
                        law_text = self.statute_manager.get_law_text(target_id)
                        
                        if law_text:
                            # Veritabanında varsa metniyle bas
                            print(f"      👉 {link['text_span']} ({target_id})")
                            clean_txt = law_text.replace('\n', ' ')
                            print(f"          📜 İÇERİK: {clean_txt[:150]}...")
                        else:
                            # Yoksa sadece atfı bas (Örn: TSHK veya HMK maddesi)
                            print(f"      🔹 {link['text_span']} (Metin veritabanında yok)")
                print("-" * 30)

if __name__ == "__main__":
    assistant = HybridLegalAssistant()
    
    # Test Senaryosu: Kiracı Tahliyesi
    case = "Kiracım 3 aydır kirayı ödemiyor. İhtarname çektim, 30 gün süre verdim ama hala ödemedi. Evden çıkarmak istiyorum."
    assistant.solve_case(case)