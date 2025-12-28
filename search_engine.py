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
from sentence_transformers import CrossEncoder

# .env dosyasını yükle
load_dotenv()

# --- DOSYA AYARLARI ---
# 1. KANUN (Statute) Dosyaları
STATUTE_FAISS   = "data/index/tbk_legal.index"
STATUTE_BM25    = "data/index/tbk_bm25.pkl"
STATUTE_MAPPING = "data/id_mapping.json"

# Target laws to load (matching build_dataset.py)
TARGET_LAWS = [
    "4721",  # TMK - Türk Medeni Kanunu
    "5237",  # TCK - Türk Ceza Kanunu
    "6102",  # TTK - Türk Ticaret Kanunu
    "4857",  # İş Kanunu
    "6098",  # TBK - Türk Borçlar Kanunu
]

# 2. EMSAL KARAR (Case) Dosyaları (Yargıtay Scraper ve Indexer çıktıları)
CASE_FAISS      = "data/index/yargitay_legal.index"
CASE_BM25       = "data/index/yargitay_bm25.pkl"
CASE_MAPPING    = "data/yargitay_id_mapping.json"
CASE_DATA       = "data/yargitay_dataset.json"

# 3. SIMILAR CASES Dosyaları
SIMILAR_CASES_FAISS   = "data/index/similar_cases_legal.index"
SIMILAR_CASES_BM25    = "data/index/similar_cases_bm25.pkl"
SIMILAR_CASES_MAPPING = "data/similar_cases_id_mapping.json"
SIMILAR_CASES_DATA    = "data/similar_cases.json"

MODEL_NAME = "KocLab-Bilkent/BERTurk-Legal"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY .env dosyasında bulunamadı!")
def load_all_statute_datasets():
    """
    Load all law datasets and merge them into a single list.
    Returns merged data list.
    """
    from pathlib import Path
    
    data_dir = Path("data")
    all_data = []
    
    for law_code in TARGET_LAWS:
        dataset_file = data_dir / f"{law_code}_dataset.json"
        
        if dataset_file.exists():
            try:
                with open(dataset_file, "r", encoding="utf-8") as f:
                    law_data = json.load(f)
                    all_data.extend(law_data)
            except Exception as e:
                print(f"   ⚠️  {dataset_file.name} yüklenirken hata: {e}")
    
    return all_data

class StatuteManager:
    """
    Bu sınıfın görevi: Yargıtay kararında geçen "LAW:6098-12" gibi bir atfı alıp,
    Tüm kanun veri setimizdeki parça parça (fıkra) metinleri birleştirerek tam maddeyi getirmektir.
    """
    def __init__(self, all_data):
        print("   -> Kanun Metinleri Yükleniyor (StatuteManager)...")
        self.articles = {} 
        self._load_data(all_data)

    def _load_data(self, data):
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

        # 2. Cross-Encoder Modeli Yükle (Two-Stage Reranking için)
        print("   -> Cross-Encoder Modeli Yükleniyor (BERTurk)...")
        try:
            # Turkish-compatible Cross-Encoder model
            # Note: If "cross-encoder/turkish-bert-base" is not available, 
            # this will fall back to a multilingual model
            try:
                self.cross_encoder = CrossEncoder("cross-encoder/turkish-bert-base")
            except:
                # Fallback to multilingual model that works well with Turkish
                self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
            print("   -> Cross-Encoder hazır (Two-Stage Reranking aktif)")
        except Exception as e:
            print(f"   UYARI: Cross-Encoder yüklenemedi ({e}). Reranking devre dışı.")
            self.cross_encoder = None

        # 3. Kanun (Statute) İndekslerini Yükle
        print("   -> Kanun İndeksleri Yükleniyor...")
        all_statute_data = None
        try:
            self.statute_index = faiss.read_index(STATUTE_FAISS)
            self.statute_ids = json.load(open(STATUTE_MAPPING, "r", encoding="utf-8"))
            self.statute_bm25_data = pickle.load(open(STATUTE_BM25, "rb"))
            # Kanun verilerini yükle - TARGET_LAWS listesindeki tüm kanunları birleştir
            all_statute_data = load_all_statute_datasets()
            # ID mapping'e göre verileri eşleştir
            self.statute_data = {}
            for chunk in all_statute_data:
                chunk_id = chunk.get('id')
                if chunk_id:
                    self.statute_data[chunk_id] = chunk
            print(f"   -> {len(self.statute_data)} kanun maddesi yüklendi")
        except Exception as e:
            print(f"⚠️  UYARI: Kanun indeksleri yüklenemedi ({e}). Kanun araması devre dışı.")
            self.statute_index = None
            self.statute_ids = None
            self.statute_bm25_data = None
            self.statute_data = {}
        
        # 4. StatuteManager'ı yükle (kanun metinlerini birleştirmek için)
        try:
            if all_statute_data is None:
                all_statute_data = load_all_statute_datasets()
            self.statute_manager = StatuteManager(all_statute_data)
        except Exception as e:
            print(f"⚠️  UYARI: StatuteManager yüklenemedi ({e})")
            self.statute_manager = None

        # 5. Similar Cases İndekslerini Yükle
        print("   -> Similar Cases İndeksleri Yükleniyor...")
        try:
            self.similar_cases_index = faiss.read_index(SIMILAR_CASES_FAISS)
            self.similar_cases_ids = json.load(open(SIMILAR_CASES_MAPPING, "r", encoding="utf-8"))
            self.similar_cases_bm25_data = pickle.load(open(SIMILAR_CASES_BM25, "rb"))
            # Similar cases verilerini yükle - ID mapping'e göre eşleştir
            similar_cases_raw = json.load(open(SIMILAR_CASES_DATA, "r", encoding="utf-8"))
            # Mapping dosyasındaki ID sırası ile verileri eşleştir
            self.similar_cases_data = {}
            for idx, case_id in enumerate(self.similar_cases_ids):
                if idx < len(similar_cases_raw):
                    self.similar_cases_data[case_id] = similar_cases_raw[idx]
            print(f"   -> {len(self.similar_cases_data)} similar case yüklendi")
        except Exception as e:
            print(f"❌ HATA: Similar cases dosyaları bulunamadı ({e})")
            raise Exception(f"Similar cases indeksleri yüklenemedi: {e}")
        
        print("✅ Sistem Hazır!\n")

    def _generate_similar_case_id(self, case, index):
        """Similar case için ID oluştur (karar_indexer.py ile uyumlu)"""
        prefix = "SIMILAR_CASES"
        if case.get('url'):
            url = case['url']
            if 'v=' in url:
                url_id = url.split('v=')[-1]
                return f"{prefix}:{url_id}"
        
        case_num = case.get('case_number', '').strip().replace(' ', '_')
        decision_num = case.get('decision_number', '').strip().replace(' ', '_')
        chamber = case.get('chamber', '').strip().replace(' ', '_').replace('.', '')
        
        if case_num and decision_num:
            return f"{prefix}:{chamber}_{case_num}_{decision_num}"
        
        return f"{prefix}:{index}"

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

    def _search_generic(self, query, faiss_index, id_map, bm25_model, bm25_ids, k=50, alpha=0.7):
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

    def _apply_authority_boost(self, results, data_dict):
        """
        Authority-Aware Ranking: Türk hukuk hiyerarşisine göre skorları artırır.
        
        Multipliers:
        - İÇTİHADI BİRLEŞTİRME / İBK: x2.0
        - HUKUK GENEL / HGK: x1.5
        - YARGITAY / \d+HD (örn: 3.HD): x1.2
        - BÖLGE ADLİYE / BAM: x1.1
        - Others (Statutes/Local): x1.0
        """
        boosted_results = []
        
        for doc_id, score in results:
            if doc_id not in data_dict:
                boosted_results.append((doc_id, score, None))
                continue
            
            data = data_dict[doc_id]
            
            # Similar cases için farklı yapı
            if 'metadata' not in data:
                # Similar case yapısı - chamber'dan boost yap
                chamber = data.get('chamber', '')
                court = 'YARGITAY' if 'YARGITAY' in chamber.upper() or 'HD' in chamber.upper() else ''
                citation_label = chamber
            else:
                # Statute veya case yapısı
                metadata = data.get('metadata', {})
                citation_label = metadata.get('poly_vector', {}).get('citation_label', '')
                hierarchy = metadata.get('hierarchy', {})
                court = hierarchy.get('court', '')
                chamber = hierarchy.get('chamber', '')
            
            # Boost multiplier belirleme
            multiplier = 1.0
            boost_tag = None
            
            # İçtihadi Birleştirme / İBK kontrolü
            if 'İÇTİHADI BİRLEŞTİRME' in citation_label.upper() or 'İBK' in citation_label.upper():
                multiplier = 2.0
                boost_tag = "İBK"
            # Hukuk Genel Kurulu / HGK kontrolü
            elif 'HUKUK GENEL' in chamber.upper() or 'HGK' in citation_label.upper() or 'HGK' in chamber.upper():
                multiplier = 1.5
                boost_tag = "HGK"
            # Yargıtay / HD (Hukuk Dairesi) kontrolü
            elif 'YARGITAY' in court.upper() or re.search(r'\d+HD', citation_label.upper()) or re.search(r'\d+\.\s*HD', chamber.upper()):
                multiplier = 1.2
                boost_tag = "YARGITAY"
            # Bölge Adliye / BAM kontrolü
            elif 'BÖLGE ADLİYE' in court.upper() or 'BAM' in citation_label.upper():
                multiplier = 1.1
                boost_tag = "BAM"
            # Diğerleri (Statutes/Local): x1.0 (değişiklik yok)
            
            boosted_score = score * multiplier
            boosted_results.append((doc_id, boosted_score, boost_tag))
        
        # Boost edilmiş skorlara göre yeniden sırala
        boosted_results.sort(key=lambda x: x[1], reverse=True)
        return boosted_results

    def _rerank_results(self, query, results, data_dict, top_k=5):
        """
        Two-Stage Reranking: Cross-Encoder kullanarak top 50 sonucu yeniden sıralar.
        
        Args:
            query: Kullanıcı sorgusu
            results: (doc_id, score, boost_tag) tuple listesi (top 50)
            data_dict: Doküman verileri
            top_k: Final olarak döndürülecek sonuç sayısı
        
        Returns:
            Rerank edilmiş (doc_id, rerank_score, boost_tag) listesi
        """
        if not self.cross_encoder or len(results) == 0:
            # Cross-Encoder yoksa veya sonuç yoksa, mevcut skorları kullan
            return [(r[0], r[1], r[2] if len(r) > 2 else None) for r in results[:top_k]]
        
        # Top 50 sonucu al
        top_50 = results[:50]
        
        # Her doküman için metin hazırla
        pairs = []
        doc_ids = []
        boost_tags = []
        
        for item in top_50:
            doc_id = item[0]
            boost_tag = item[2] if len(item) > 2 else None
            
            if doc_id not in data_dict:
                continue
            
            data = data_dict[doc_id]
            # Doküman metnini hazırla: metadata yapısına göre
            if 'metadata' in data:
                # Statute veya case yapısı
                summary = data.get('metadata', {}).get('sac_context', {}).get('parent_summary', '')
                text = data.get('text', '')
                doc_text = f"{summary} {text}".strip()
            else:
                # Similar case yapısı (direkt full_text var)
                full_text = data.get('full_text', '')
                if not full_text:
                    summary = data.get('summary', '') or ''
                    conclusion = data.get('conclusion', '') or ''
                    full_text = f"{summary} {conclusion}".strip()
                doc_text = full_text
            
            pairs.append([query, doc_text])
            doc_ids.append(doc_id)
            boost_tags.append(boost_tag)
        
        if not pairs:
            return []
        
        # Cross-Encoder ile skorları hesapla
        try:
            rerank_scores = self.cross_encoder.predict(pairs)
            rerank_scores = rerank_scores.tolist() if hasattr(rerank_scores, 'tolist') else list(rerank_scores)
        except Exception as e:
            print(f"   UYARI: Cross-Encoder reranking hatası ({e}). Orijinal skorlar kullanılıyor.")
            return [(r[0], r[1], r[2] if len(r) > 2 else None) for r in top_50[:top_k]]
        
        # Rerank skorlarına göre sırala
        reranked = list(zip(doc_ids, rerank_scores, boost_tags))
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked[:top_k]

    def search_similar_cases(self, query, top_k=5):
        """
        Search for similar cases and return top-k results.
        Returns a list of (doc_id, score, boost_tag) tuples.
        """
        if not self.similar_cases_index:
            return []
        
        # Generate legal query
        legal_query = self._generate_legal_query(query)
        
        # Stage 1: Recall - Get Top 50 candidates
        similar_case_results = self._search_generic(
            legal_query, self.similar_cases_index, self.similar_cases_ids,
            self.similar_cases_bm25_data["model"], self.similar_cases_bm25_data["ids"], k=50
        )
        
        # Stage 2: Authority Boost (similar cases için de uygulanabilir)
        boosted_similar_case_results = self._apply_authority_boost(similar_case_results, self.similar_cases_data)
        
        # Stage 3: Rerank with Cross-Encoder (ŞİMDİLİK DEVRE DIŞI)
        # final_similar_case_results = self._rerank_results(
        #     legal_query, boosted_similar_case_results, self.similar_cases_data, top_k=top_k
        # )
        # Reranking olmadan direkt top_k sonucu döndür
        final_similar_case_results = boosted_similar_case_results[:top_k]
        
        return final_similar_case_results

    def search_statutes(self, query, top_k=5):
        """
        Search for relevant statutes/laws and return top-k results.
        Returns a list of (doc_id, score, boost_tag) tuples.
        """
        if not self.statute_index or not self.statute_ids or not self.statute_bm25_data:
            return []
        
        # Generate legal query
        legal_query = self._generate_legal_query(query)
        
        # Stage 1: Recall - Get Top 50 candidates
        statute_results = self._search_generic(
            legal_query, self.statute_index, self.statute_ids,
            self.statute_bm25_data["model"], self.statute_bm25_data["ids"], k=50
        )
        
        # Stage 2: Authority Boost
        boosted_statute_results = self._apply_authority_boost(statute_results, self.statute_data)
        
        # Stage 3: Rerank with Cross-Encoder (ŞİMDİLİK DEVRE DIŞI)
        # final_statute_results = self._rerank_results(
        #     legal_query, boosted_statute_results, self.statute_data, top_k=top_k
        # )
        # Reranking olmadan direkt top_k sonucu döndür
        final_statute_results = boosted_statute_results[:top_k]
        
        return final_statute_results

    def search(self, query, top_k=5):
        """
        Ana arama fonksiyonu: Hem similar cases hem de kanun sonuçlarını döndürür.
        Returns: (similar_case_results, statute_results)
        """
        final_similar_case_results = self.search_similar_cases(query, top_k=top_k)
        final_statute_results = self.search_statutes(query, top_k=top_k)
        return final_similar_case_results, final_statute_results

    def solve_case(self, user_case):
        print("="*60)
        print(f"📝 OLAY: {user_case[:100]}...")
        
        # 1. Query Generation
        legal_query = self._generate_legal_query(user_case)
        print(f"🔍 HUKUKİ SORGU: {legal_query}")
        print("="*60)

        # --- BÖLÜM 1: İLGİLİ KANUN MADDELERİ (Two-Stage Pipeline) ---
        print("\n" + "⚖️  İLGİLİ KANUN MADDELERİ (MEVZUAT)".center(60, "-"))
        
        # Stage 1: Recall - Get Top 50 candidates
        print("   [1/2] Recall: Top 50 aday belirleniyor...")
        statute_results = self._search_generic(
            legal_query, self.statute_index, self.statute_ids, 
            self.statute_bm25_data["model"], self.statute_bm25_data["ids"], k=50
        )
        
        # Stage 2: Authority Boost
        print("   [2/2] Authority-Aware Ranking uygulanıyor...")
        boosted_statute_results = self._apply_authority_boost(statute_results, self.statute_data)
        
        # Stage 3: Rerank with Cross-Encoder (ŞİMDİLİK DEVRE DIŞI)
        # print("   [3/3] Two-Stage Reranking (Cross-Encoder) uygulanıyor...")
        # final_statute_results = self._rerank_results(legal_query, boosted_statute_results, self.statute_data, top_k=5)
        final_statute_results = boosted_statute_results[:5]
        
        # Final Output: Top 5
        for rank, item in enumerate(final_statute_results):
            doc_id = item[0]
            rerank_score = item[1]
            boost_tag = item[2] if len(item) > 2 else None
            
            data = self.statute_data[doc_id]
            cit = data['metadata']['poly_vector']['citation_label']
            
            # Boost tag gösterimi
            boost_str = f" [Authority Boost: {boost_tag}]" if boost_tag else ""
            
            print(f"\n#{rank+1} [{cit}] (Skor: {rerank_score:.4f}){boost_str}")
            print(f"   ÖZET: {data['metadata']['sac_context']['parent_summary']}")
            print(f"   METİN: {data['text'][:150]}...")

        # --- BÖLÜM 2: EMSAL KARARLAR (Two-Stage Pipeline) ---
        if self.case_index:
            print("\n" + "🏛️  EMSAL YARGITAY KARARLARI".center(60, "-"))
            
            # Stage 1: Recall - Get Top 50 candidates
            print("   [1/2] Recall: Top 50 aday belirleniyor...")
            case_results = self._search_generic(
                legal_query, self.case_index, self.case_ids,
                self.case_bm25_data["model"], self.case_bm25_data["ids"], k=50
            )
            
            # Stage 2: Authority Boost
            print("   [2/2] Authority-Aware Ranking uygulanıyor...")
            boosted_case_results = self._apply_authority_boost(case_results, self.case_data)
            
            # Stage 3: Rerank with Cross-Encoder (ŞİMDİLİK DEVRE DIŞI)
            # print("   [3/3] Two-Stage Reranking (Cross-Encoder) uygulanıyor...")
            # final_case_results = self._rerank_results(legal_query, boosted_case_results, self.case_data, top_k=5)
            final_case_results = boosted_case_results[:5]
            
            # Final Output: Top 5
            for rank, item in enumerate(final_case_results):
                doc_id = item[0]
                rerank_score = item[1]
                boost_tag = item[2] if len(item) > 2 else None
                
                data = self.case_data[doc_id]
                cit = data['metadata']['poly_vector']['citation_label']
                
                # Boost tag gösterimi
                boost_str = f" [Authority Boost: {boost_tag}]" if boost_tag else ""
                
                print(f"\n#{rank+1} [{cit}] (Skor: {rerank_score:.4f}){boost_str}")
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

def display_results(similar_case_results, statute_results, top_k, assistant):
    """Arama sonuçlarını güzel bir şekilde göster - hem benzer kararlar hem kanunlar"""
    print("\n" + "="*80)
    print("🔍 ARAMA SONUÇLARI - BENZER KARARLAR VE KANUNLAR")
    print("="*80)
    
    # Similar Cases Sonuçları
    print(f"\n🏛️  BENZER KARARLAR (Top {min(top_k, len(similar_case_results))})")
    print("-"*80)
    if similar_case_results:
        for rank, item in enumerate(similar_case_results[:top_k], 1):
            doc_id = item[0]
            rerank_score = item[1]
            boost_tag = item[2] if len(item) > 2 else None
            
            if doc_id not in assistant.similar_cases_data:
                continue
                
            data = assistant.similar_cases_data[doc_id]
            
            # Citation oluştur
            chamber = data.get('chamber', '')
            case_num = data.get('case_number', '')
            decision_num = data.get('decision_number', '')
            decision_date = data.get('decision_date', '')
            cit = f"{chamber}, E. {case_num}, K. {decision_num}, T. {decision_date}" if case_num else doc_id
            
            boost_str = f" [Authority Boost: {boost_tag}]" if boost_tag else ""
            
            print(f"\n#{rank} [{cit}] (Skor: {rerank_score:.4f}){boost_str}")
            
            # Full text göster
            full_text = data.get('full_text', '')
            if full_text:
                print(f"   📄 KARAR: {full_text[:300]}...")
            else:
                summary = data.get('summary', '')
                conclusion = data.get('conclusion', '')
                if summary or conclusion:
                    print(f"   📄 ÖZET: {summary[:200]}...")
                    print(f"   📄 SONUÇ: {conclusion[:200]}...")
            
            # URL
            url = data.get('url', '')
            if url:
                print(f"   🔗 URL: {url}")
    else:
        print("   Sonuç bulunamadı.")
    
    # Kanun Sonuçları
    print(f"\n⚖️  UYGULANABİLECEK KANUNLAR (Top {min(top_k, len(statute_results))})")
    print("-"*80)
    if statute_results:
        for rank, item in enumerate(statute_results[:top_k], 1):
            doc_id = item[0]
            rerank_score = item[1]
            boost_tag = item[2] if len(item) > 2 else None
            
            if doc_id not in assistant.statute_data:
                continue
                
            data = assistant.statute_data[doc_id]
            
            # Citation oluştur
            cit = data.get('metadata', {}).get('poly_vector', {}).get('citation_label', doc_id)
            
            boost_str = f" [Authority Boost: {boost_tag}]" if boost_tag else ""
            
            print(f"\n#{rank} [{cit}] (Skor: {rerank_score:.4f}){boost_str}")
            
            # Özet ve metin göster
            summary = data.get('metadata', {}).get('sac_context', {}).get('parent_summary', '')
            text = data.get('text', '')
            if summary:
                print(f"   📄 ÖZET: {summary[:200]}...")
            if text:
                print(f"   📜 METİN: {text[:200]}...")
    else:
        print("   Sonuç bulunamadı.")
    
    print("\n" + "="*80)

def interactive_terminal():
    """Interactive terminal arayüzü"""
    print("\n" + "="*80)
    print("🚀 HUKUKİ ARAMA MOTORU - INTERACTIVE TERMINAL")
    print("="*80)
    print("\nKomutlar:")
    print("  - Sorgu yazın ve Enter'a basın")
    print("  - 'quit' veya 'exit' yazarak çıkış yapın")
    print("  - 'help' yazarak yardım alın")
    print("="*80 + "\n")
    
    assistant = HybridLegalAssistant()
    
    while True:
        try:
            # Query al
            query = input("\n🔍 Sorgu giriniz (veya 'quit' ile çıkış): ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Çıkış yapılıyor...")
                break
            
            if query.lower() == 'help':
                print("\n📖 YARDIM:")
                print("  - Sorgu yazın: Örn: 'Kiracı tahliyesi', 'Boşanma davası', 'İş kazası tazminat'")
                print("  - Top k sayısı: Kaç sonuç görmek istediğinizi belirtin (varsayılan: 5)")
                print("  - 'quit' veya 'exit': Programdan çıkış")
                continue
            
            # Top k al
            top_k_input = input("📊 Kaç sonuç gösterilsin? (varsayılan: 5): ").strip()
            try:
                top_k = int(top_k_input) if top_k_input else 5
                if top_k < 1:
                    top_k = 5
                if top_k > 50:
                    print("   ⚠️  Maksimum 50 sonuç gösterilebilir. 50 olarak ayarlandı.")
                    top_k = 50
            except ValueError:
                top_k = 5
                print("   ⚠️  Geçersiz sayı. Varsayılan 5 kullanılıyor.")
            
            # Arama yap
            print(f"\n🔍 Aranıyor... (Top {top_k} sonuç)")
            similar_case_results, statute_results = assistant.search(query, top_k=top_k)
            
            # Sonuçları göster
            display_results(similar_case_results, statute_results, top_k, assistant)
            
        except KeyboardInterrupt:
            print("\n\n👋 Çıkış yapılıyor...")
            break
        except Exception as e:
            print(f"\n❌ Hata oluştu: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    interactive_terminal()