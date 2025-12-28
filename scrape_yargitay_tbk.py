import json
import re
import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# --- YAPILANDIRMA ---
TARGET_COUNT = 100  # Minimum 100 karar hedefi
OUTPUT_FILE = "data/yargitay_dataset.json"
LAW_CODE = "6098"

# Kanun adları ve kısaltmaları için sözlük
LAW_NAMES = {
    "6098": "Türk Borçlar Kanunu (TBK)",
    "6101": "TBK Yürürlük Kanunu",
    "6102": "Türk Ticaret Kanunu (TTK)",
    "6100": "Hukuk Muhakemeleri Kanunu (HMK)",
    "6217": "Yargı Hizmetleri Kanunu",
    "6353": "Bazı Kanunlarda Değişiklik Kanunu",
    "5362": "Esnaf ve Sanatkarlar Meslek Kuruluşları Kanunu",
    "2004": "İcra ve İflas Kanunu (İİK)",
    "1086": "Hukuk Usulü Muhakemeleri Kanunu (HUMK)",
    "213": "Vergi Usul Kanunu (VUK)",
    "4721": "Türk Medeni Kanunu (TMK)",
    "5237": "Türk Ceza Kanunu (TCK)",
}

# Kısaltma -> Kanun No eşleştirmesi
LAW_ABBREV = {
    "TBK": "6098",
    "TTK": "6102",
    "TMK": "4721",
    "TCK": "5237",
    "HMK": "6100",
    "İİK": "2004",
    "IIK": "2004",
    "HUMK": "1086",
    "VUK": "213",
    "BK": "818",  # Eski Borçlar Kanunu
}

class YargitayScraperV4:
    def __init__(self):
        options = webdriver.ChromeOptions()
        # options.add_argument('--headless')  # Görmek için kapalı
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--window-size=1920,1080')
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        self.driver.implicitly_wait(10)
        self.wait = WebDriverWait(self.driver, 15)
        self.cases = []
        self.collected_count = 0
        self.seen_ids = set()  # Tekrar eden kararları önle

    def start(self):
        url = "https://karararama.yargitay.gov.tr/"
        self.driver.get(url)
        print("=" * 60)
        print("YARGITAY KARAR SCRAPER V4")
        print("=" * 60)
        print("\n📋 TALİMATLAR:")
        print("1. Arama kutusuna '6098' yazın ve 'Ara' butonuna tıklayın")
        print("2. Eğer CAPTCHA çıkarsa çözün")
        print("3. Sonuçlar listesi göründüğünde ENTER'a basın")
        print("4. İlk satıra tıklayarak sağ panelde detayın açıldığından emin olun")
        print("\n⚠️  ÖNEMLİ: Sağ panelde 'İçtihat Metni' başlığı görünmeli!")
        input("\n👉 Hazır olduğunuzda ENTER'a basın...")
        
        self._scrape_loop()

    def _scrape_loop(self):
        page = 1
        consecutive_failures = 0
        
        while self.collected_count < TARGET_COUNT and consecutive_failures < 5:
            print(f"\n{'='*50}")
            print(f"📄 Sayfa {page} taranıyor... (Toplam: {self.collected_count})")
            print('='*50)
            
            # Sayfanın yüklenmesini bekle
            time.sleep(2)
            
            # Tablo satırlarını bul
            rows = self._get_table_rows()
            
            if not rows:
                print("⚠️ Tablo satırı bulunamadı!")
                consecutive_failures += 1
                continue
            
            print(f"   -> {len(rows)} satır bulundu.")
            
            added_on_page = 0
            
            for idx in range(len(rows)):
                if self.collected_count >= TARGET_COUNT:
                    break
                
                try:
                    # Satırları her seferinde yeniden al (DOM değişebilir)
                    rows = self._get_table_rows()
                    if idx >= len(rows):
                        break
                    
                    row = rows[idx]
                    
                    # Satır bilgilerini al
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) < 5:
                        continue
                    
                    sira_no = cells[0].text.strip()
                    daire = cells[1].text.strip()
                    esas_no = cells[2].text.strip()
                    karar_no = cells[3].text.strip()
                    karar_tarihi = cells[4].text.strip()
                    
                    # Benzersiz ID kontrolü
                    unique_key = f"{daire}-{esas_no}-{karar_no}"
                    if unique_key in self.seen_ids:
                        continue
                    
                    print(f"\n   [{idx+1}/{len(rows)}] {daire} E.{esas_no} K.{karar_no}")
                    
                    # Satıra tıkla
                    self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", row)
                    time.sleep(0.3)
                    row.click()
                    time.sleep(2)  # Detay yüklensin
                    
                    # Detay metnini al
                    detail_text = self._get_detail_text()
                    
                    if detail_text and len(detail_text) > 200:
                        print(f"      📝 Metin uzunluğu: {len(detail_text)} karakter")
                        
                        # İşle ve kaydet
                        chunks = self._process_decision(detail_text, daire, esas_no, karar_no, karar_tarihi)
                        if chunks:
                            for chunk in chunks:
                                # Graph links'i göster
                                links_count = len(chunk['metadata']['graph_links'])
                                print(f"      🔗 {links_count} kanun/madde referansı bulundu")
                            
                            self.cases.extend(chunks)
                            self.collected_count += len(chunks)
                            self.seen_ids.add(unique_key)
                            added_on_page += 1
                            print(f"      ✅ Karar #{self.collected_count} eklendi")
                    else:
                        print(f"      ⚠️ Detay metni alınamadı veya çok kısa")
                    
                except Exception as e:
                    print(f"      ❌ Hata: {str(e)[:80]}")
                    continue
            
            print(f"\n✅ Sayfadan {added_on_page} karar eklendi.")
            self._save_json()
            
            if added_on_page == 0:
                consecutive_failures += 1
                print(f"⚠️ Ardışık başarısızlık: {consecutive_failures}/5")
            else:
                consecutive_failures = 0
            
            if self.collected_count >= TARGET_COUNT:
                break
            
            # Sonraki sayfaya git
            if not self._go_next_page():
                print("Sonraki sayfa yok veya bitti.")
                break
            
            page += 1
            time.sleep(random.uniform(2, 4))
        
        print(f"\n{'='*60}")
        print(f"🎉 TAMAMLANDI! Toplam {self.collected_count} karar toplandı!")
        print(f"{'='*60}")

    def _get_table_rows(self):
        """Tablo satırlarını bul"""
        try:
            # Farklı seçiciler dene
            selectors = [
                "table.dataTable tbody tr",
                "#resultTable tbody tr",
                ".dataTables_scrollBody table tbody tr",
                "table tbody tr[role='row']",
                "table tbody tr"
            ]
            
            for selector in selectors:
                rows = self.driver.find_elements(By.CSS_SELECTOR, selector)
                # Boş olmayan satırları filtrele
                valid_rows = [r for r in rows if len(r.find_elements(By.TAG_NAME, "td")) >= 5]
                if valid_rows:
                    return valid_rows
            
            return []
        except:
            return []

    def _get_detail_text(self):
        """Sağ paneldeki karar detay metnini al - GELİŞTİRİLMİŞ"""
        
        # Yöntem 1: JavaScript ile doğrudan içerik al
        try:
            # İçtihat metni içeren elementi bul
            script = """
            // İçtihat Metni başlığını bul
            var headers = document.querySelectorAll('*');
            for (var i = 0; i < headers.length; i++) {
                var el = headers[i];
                if (el.textContent && el.textContent.includes('İçtihat Metni')) {
                    // Bu elementin parent'ını veya sonraki kardeşini al
                    var parent = el.closest('.card-body') || el.closest('.karar-content') || el.parentElement;
                    if (parent) {
                        return parent.innerText;
                    }
                }
            }
            
            // Fallback: En uzun metin içeren card-body'yi bul
            var cards = document.querySelectorAll('.card-body, [class*="karar"], [class*="detail"], [class*="content"]');
            var longestText = '';
            for (var j = 0; j < cards.length; j++) {
                var text = cards[j].innerText;
                if (text.length > longestText.length && 
                    (text.includes('Mahkeme') || text.includes('karar') || text.includes('Dairesi'))) {
                    longestText = text;
                }
            }
            return longestText;
            """
            result = self.driver.execute_script(script)
            if result and len(result) > 200 and ('Mahkeme' in result or 'karar' in result.lower() or 'İçtihat' in result):
                return result
        except Exception as e:
            print(f"      JS Yöntem 1 hatası: {e}")
        
        # Yöntem 2: BeautifulSoup ile parse et
        try:
            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            
            # "İçtihat Metni" başlığını bul
            ictihat_header = soup.find(string=re.compile(r'İçtihat\s*Metni', re.IGNORECASE))
            if ictihat_header:
                # Başlığın parent elementini bul ve sonraki içeriği al
                parent = ictihat_header.find_parent(['div', 'section', 'article'])
                if parent:
                    text = parent.get_text(separator='\n', strip=True)
                    if len(text) > 200:
                        return text
            
            # Sağ paneli bul (col-md-6, col-6 vb.)
            right_panels = soup.select('.col-md-6:last-child, .col-6:last-child, .col-lg-6:last-child')
            for panel in right_panels:
                text = panel.get_text(separator='\n', strip=True)
                if len(text) > 500 and ('Mahkeme' in text or 'karar' in text.lower()):
                    return text
            
            # card-body elementlerini kontrol et
            cards = soup.select('.card-body')
            for card in cards:
                text = card.get_text(separator='\n', strip=True)
                # İçtihat metni içeriyorsa al
                if 'İçtihat Metni' in text and len(text) > 500:
                    return text
            
            # Fallback: En uzun içerik
            all_divs = soup.find_all(['div', 'article', 'section'])
            longest = ""
            for div in all_divs:
                text = div.get_text(separator='\n', strip=True)
                if len(text) > len(longest):
                    # Form elementlerini filtrele
                    if not any(x in text for x in ['Esas yılı', 'Karar yılı', 'Büyükten Küçüğe']):
                        if 'Mahkeme' in text or 'Dairesi' in text or 'karar' in text.lower():
                            longest = text
            
            if len(longest) > 300:
                return longest
                
        except Exception as e:
            print(f"      BS Yöntem 2 hatası: {e}")
        
        # Yöntem 3: Specific element ID/class
        try:
            specific_selectors = [
                "#karar-detay",
                "#karar-metni", 
                ".karar-icerik",
                ".karar-text",
                "[data-karar]",
                ".sonuc-detay"
            ]
            for selector in specific_selectors:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for el in elements:
                    text = el.text
                    if len(text) > 300:
                        return text
        except:
            pass
        
        return None

    def _process_decision(self, text, daire, esas_no, karar_no, karar_tarihi):
        """Karar metnini işle ve JSON formatına dönüştür"""
        
        # Daire kodunu oluştur
        daire_code = self._get_daire_code(daire)
        
        # Bölümleri ayır
        sections = self._split_sections(text)
        
        if not sections:
            # Eğer bölüm bulunamazsa tüm metni al
            if len(text) > 100:
                sections = [("TAM_METIN", text)]
        
        processed = []
        for sec_name, content in sections:
            if len(content) < 50:
                continue
            
            # ID oluştur
            clean_esas = esas_no.replace('/', '-')
            clean_karar = karar_no.replace('/', '-') if karar_no else "NO-KARAR"
            chunk_id = f"CASE:{daire_code}-{clean_esas}-{clean_karar}-{sec_name}"
            
            # Tekrar kontrolü
            if chunk_id in self.seen_ids:
                continue
            
            # Graph Links - TÜM kanun/madde referanslarını çıkar
            links = self._extract_all_law_references(content)
            
            # Citation oluştur
            citation = f"Yargıtay {daire_code} E.{esas_no}"
            
            # Özet
            summary = f"Bu karar, {daire} tarafından {karar_tarihi} tarihinde verilmiştir."
            
            obj = {
                "id": chunk_id,
                "text": content.replace('\n', ' ').strip(),
                "metadata": {
                    "doc_type": "decision",
                    "hierarchy": {
                        "court": "Yargıtay",
                        "chamber": daire,
                        "basis_no": esas_no,
                        "decision_no": karar_no,
                        "section": sec_name
                    },
                    "dates": {"decision_date": karar_tarihi},
                    "urls": {"source_url": "https://karararama.yargitay.gov.tr/"},
                    "sac_context": {"parent_summary": summary},
                    "poly_vector": {
                        "citation_label": citation,
                        "citation_variations": [
                            citation,
                            f"{daire} {esas_no}",
                            f"E: {esas_no} K: {karar_no}"
                        ]
                    },
                    "graph_links": links
                }
            }
            processed.append(obj)
        
        return processed

    def _get_daire_code(self, daire):
        """Daire adından kısa kod oluştur"""
        match = re.search(r"(\d+)\.\s*Hukuk\s*Daire", daire, re.IGNORECASE)
        if match:
            return f"{match.group(1)}HD"
        
        match = re.search(r"(\d+)\.\s*Ceza\s*Daire", daire, re.IGNORECASE)
        if match:
            return f"{match.group(1)}CD"
        
        if "Hukuk Genel" in daire:
            return "HGK"
        if "Ceza Genel" in daire:
            return "CGK"
        
        return "YARGITAY"

    def _split_sections(self, text):
        """Metni mantıksal bölümlere ayır"""
        sections = []
        
        # Gerekçe ve Hüküm başlıklarını bul
        gerekce_match = re.search(r"(?i)(?:[IVX]+\.\s*)?G\s*E\s*R\s*E\s*K\s*Ç\s*E", text)
        hukum_match = re.search(r"(?i)(?:[IVX]+\.\s*)?(?:H\s*Ü\s*K\s*Ü\s*M|S\s*O\s*N\s*U\s*Ç)", text)
        
        if gerekce_match and hukum_match and gerekce_match.start() < hukum_match.start():
            gerekce_text = text[gerekce_match.end():hukum_match.start()].strip()
            hukum_text = text[hukum_match.end():].strip()
            
            if len(gerekce_text) > 50:
                sections.append(("GEREKCE", gerekce_text))
            if len(hukum_text) > 50:
                sections.append(("HUKUM", hukum_text))
        elif gerekce_match:
            sections.append(("GEREKCE", text[gerekce_match.end():].strip()))
        elif hukum_match:
            sections.append(("HUKUM", text[hukum_match.end():].strip()))
        
        # Başlık yoksa İçtihat Metni'nden sonrasını al
        if not sections:
            ictihat_match = re.search(r"İçtihat\s*Metni", text, re.IGNORECASE)
            if ictihat_match:
                clean_text = text[ictihat_match.end():].strip()
                if len(clean_text) > 100:
                    sections.append(("TAM_METIN", clean_text))
            elif len(text) > 300:
                sections.append(("TAM_METIN", text))
        
        return sections

    def _extract_all_law_references(self, text):
        """Metindeki TÜM kanun ve madde atıflarını çıkar - KAPSAMLI"""
        links = []
        seen = set()
        
        # ========================================================
        # PATTERN 1: "XXXX Sayılı Kanun/Yasa'nın XX. maddesi"
        # Örnek: "6098 Sayılı Türk Borçlar Kanununun 346.maddesinde"
        # ========================================================
        pattern1 = r"(\d{3,5})\s*[Ss]ayılı\s*[^,\.;]{0,50}?(\d+)[\.\s']*(?:(?:ı|i|u|ü)?nc(?:ı|i|u|ü)?\s*)?[Mm]adde"
        for match in re.finditer(pattern1, text):
            law_no = match.group(1)
            article_no = match.group(2)
            target_id = f"LAW:{law_no}-{article_no}"
            if target_id not in seen:
                links.append({
                    "target_id": target_id,
                    "relation_type": "applies_statute",
                    "text_span": match.group(0)[:80],
                    "law_name": LAW_NAMES.get(law_no, f"{law_no} Sayılı Kanun")
                })
                seen.add(target_id)
        
        # ========================================================
        # PATTERN 2: "(XXXX) Madde YYY" formatı
        # Örnek: "TÜRK BORÇLAR KANUNU (6098) Madde 346"
        # ========================================================
        pattern2 = r"\((\d{3,5})\)\s*[Mm]adde\s*(\d+)"
        for match in re.finditer(pattern2, text):
            law_no = match.group(1)
            article_no = match.group(2)
            target_id = f"LAW:{law_no}-{article_no}"
            if target_id not in seen:
                links.append({
                    "target_id": target_id,
                    "relation_type": "applies_statute",
                    "text_span": match.group(0),
                    "law_name": LAW_NAMES.get(law_no, f"{law_no} Sayılı Kanun")
                })
                seen.add(target_id)
        
        # ========================================================
        # PATTERN 3: Kısaltma formatları
        # Örnek: "TBK.nun 346.maddesi", "TBK'nın 346'ncı maddesi"
        # ========================================================
        pattern3 = r"(TBK|TTK|TMK|TCK|HMK|İİK|IIK|HUMK|VUK|BK)['\.\s]*(?:n[uıiü]n|ya|ye|nda|nde|dan|den)?\s*(\d+)[\.\s']*(?:(?:ı|i|u|ü)?nc(?:ı|i|u|ü)?\s*)?[Mm]adde"
        for match in re.finditer(pattern3, text, re.IGNORECASE):
            abbrev = match.group(1).upper()
            article_no = match.group(2)
            law_no = LAW_ABBREV.get(abbrev, "")
            if law_no:
                target_id = f"LAW:{law_no}-{article_no}"
                if target_id not in seen:
                    links.append({
                        "target_id": target_id,
                        "relation_type": "applies_statute",
                        "text_span": match.group(0),
                        "law_name": LAW_NAMES.get(law_no, abbrev)
                    })
                    seen.add(target_id)
        
        # ========================================================
        # PATTERN 4: Sadece kanun numarası (madde olmadan)
        # Örnek: "6098 sayılı Kanun", "6098 sayılı Yasa"
        # ========================================================
        pattern4 = r"(\d{3,5})\s*[Ss]ayılı\s*(?:Kanun|Yasa|Türk\s+Borçlar\s+Kanunu)"
        for match in re.finditer(pattern4, text, re.IGNORECASE):
            law_no = match.group(1)
            target_id = f"LAW:{law_no}"
            if target_id not in seen:
                links.append({
                    "target_id": target_id,
                    "relation_type": "references_statute",
                    "text_span": match.group(0),
                    "law_name": LAW_NAMES.get(law_no, f"{law_no} Sayılı Kanun")
                })
                seen.add(target_id)
        
        # ========================================================
        # PATTERN 5: Tek başına madde numarası (context'ten kanun bul)
        # Örnek: "76'ncı maddesi", "88 inci maddesinin"
        # ========================================================
        pattern5 = r"(\d{1,3})[\s']*(?:'?(?:ı|i|u|ü)?nc(?:ı|i|u|ü)?)\s*[Mm]adde"
        for match in re.finditer(pattern5, text, re.IGNORECASE):
            article_no = match.group(1)
            # Yakın bağlamda kanun numarası ara
            context_start = max(0, match.start() - 300)
            context = text[context_start:match.start()]
            law_match = re.search(r"(\d{3,5})\s*[Ss]ayılı", context)
            if law_match:
                law_no = law_match.group(1)
                target_id = f"LAW:{law_no}-{article_no}"
                if target_id not in seen:
                    links.append({
                        "target_id": target_id,
                        "relation_type": "applies_statute",
                        "text_span": match.group(0),
                        "law_name": LAW_NAMES.get(law_no, f"{law_no} Sayılı Kanun")
                    })
                    seen.add(target_id)
        
        # ========================================================
        # PATTERN 6: Çoklu madde listesi
        # Örnek: "323, 325, 331, 340, 343, 344, 346 ve 354'ncü maddelerinin"
        # ========================================================
        pattern6 = r"(\d{1,3}(?:\s*,\s*\d{1,3})+(?:\s*ve\s*\d{1,3})?)[\s']*(?:'?(?:ı|i|u|ü)?nc(?:ı|i|u|ü)?\s*)?[Mm]adde"
        for match in re.finditer(pattern6, text, re.IGNORECASE):
            articles_str = match.group(1)
            articles = re.findall(r"\d+", articles_str)
            
            # Bağlamda kanun numarası bul
            context_start = max(0, match.start() - 400)
            context = text[context_start:match.start()]
            law_match = re.search(r"(\d{3,5})\s*[Ss]ayılı", context)
            law_no = law_match.group(1) if law_match else "6098"
            
            for article_no in articles:
                if int(article_no) > 1000:  # Kanun numarası değil, madde numarası olmalı
                    continue
                target_id = f"LAW:{law_no}-{article_no}"
                if target_id not in seen:
                    links.append({
                        "target_id": target_id,
                        "relation_type": "applies_statute",
                        "text_span": f"{law_no} m.{article_no}",
                        "law_name": LAW_NAMES.get(law_no, f"{law_no} Sayılı Kanun")
                    })
                    seen.add(target_id)
        
        # ========================================================
        # PATTERN 7: İsimle kanun atfı
        # Örnek: "Türk Borçlar Kanunu'nun 346. maddesi"
        # ========================================================
        kanun_patterns = [
            (r"Türk\s+Borçlar\s+Kanunu['\s]*(?:n[uı]n)?\s*(\d+)", "6098"),
            (r"Türk\s+Ticaret\s+Kanunu['\s]*(?:n[uı]n)?\s*(\d+)", "6102"),
            (r"Türk\s+Medeni\s+Kanunu['\s]*(?:n[uı]n)?\s*(\d+)", "4721"),
            (r"Türk\s+Ceza\s+Kanunu['\s]*(?:n[uı]n)?\s*(\d+)", "5237"),
            (r"Hukuk\s+Muhakemeleri\s+Kanunu['\s]*(?:n[uı]n)?\s*(\d+)", "6100"),
            (r"İcra\s+ve\s+İflas\s+Kanunu['\s]*(?:n[uı]n)?\s*(\d+)", "2004"),
            (r"Vergi\s+Usul\s+Kanunu['\s]*(?:n[uı]n)?\s*(\d+)", "213"),
            (r"Borçlar\s+Kanunu['\s]*(?:n[uı]n)?\s*(\d+)", "6098"),
            (r"Ticaret\s+Kanunu['\s]*(?:n[uı]n)?\s*(\d+)", "6102"),
        ]
        
        for pattern, law_no in kanun_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                article_no = match.group(1)
                target_id = f"LAW:{law_no}-{article_no}"
                if target_id not in seen:
                    links.append({
                        "target_id": target_id,
                        "relation_type": "applies_statute",
                        "text_span": match.group(0)[:60],
                        "law_name": LAW_NAMES.get(law_no, "")
                    })
                    seen.add(target_id)
        
        # ========================================================
        # PATTERN 8: Geçici madde referansları
        # Örnek: "geçici 2.maddesi", "geçici madde 3"
        # ========================================================
        pattern8 = r"[Gg]eçici\s*(?:(\d+)[\.\s]*)?[Mm]adde(?:\s*(\d+))?"
        for match in re.finditer(pattern8, text):
            article_no = match.group(1) or match.group(2)
            if article_no:
                context_start = max(0, match.start() - 200)
                context = text[context_start:match.start()]
                law_match = re.search(r"(\d{3,5})\s*[Ss]ayılı", context)
                law_no = law_match.group(1) if law_match else "6098"
                
                target_id = f"LAW:{law_no}-gecici{article_no}"
                if target_id not in seen:
                    links.append({
                        "target_id": target_id,
                        "relation_type": "applies_statute",
                        "text_span": match.group(0),
                        "law_name": LAW_NAMES.get(law_no, f"{law_no} Sayılı Kanun")
                    })
                    seen.add(target_id)
        
        # ========================================================
        # PATTERN 9: Fıkra referansları
        # Örnek: "177.maddesinin 1.fıkrası"
        # ========================================================
        pattern9 = r"(\d+)[\.\s]*[Mm]adde(?:si)?(?:n[iı]n)?\s*(\d+)[\.\s]*[Ff]ıkra"
        for match in re.finditer(pattern9, text, re.IGNORECASE):
            article_no = match.group(1)
            fikra_no = match.group(2)
            
            context_start = max(0, match.start() - 300)
            context = text[context_start:match.start()]
            law_match = re.search(r"(\d{3,5})\s*[Ss]ayılı", context)
            if law_match:
                law_no = law_match.group(1)
                target_id = f"LAW:{law_no}-{article_no}-f{fikra_no}"
                if target_id not in seen:
                    links.append({
                        "target_id": target_id,
                        "relation_type": "applies_statute",
                        "text_span": match.group(0),
                        "law_name": LAW_NAMES.get(law_no, f"{law_no} Sayılı Kanun")
                    })
                    seen.add(target_id)
        
        # ========================================================
        # PATTERN 10: Basit madde referansları
        # Örnek: "m. 346", "md. 346", "Madde 346"
        # ========================================================
        pattern10 = r"(?:m\.|md\.|[Mm]adde)\s*(\d+)"
        for match in re.finditer(pattern10, text):
            article_no = match.group(1)
            if int(article_no) > 1000:  # Kanun numarası değil
                continue
            
            context_start = max(0, match.start() - 250)
            context = text[context_start:match.start()]
            law_match = re.search(r"(\d{3,5})\s*[Ss]ayılı", context)
            if law_match:
                law_no = law_match.group(1)
                target_id = f"LAW:{law_no}-{article_no}"
                if target_id not in seen:
                    links.append({
                        "target_id": target_id,
                        "relation_type": "applies_statute",
                        "text_span": match.group(0),
                        "law_name": LAW_NAMES.get(law_no, f"{law_no} Sayılı Kanun")
                    })
                    seen.add(target_id)
        
        return links

    def _go_next_page(self):
        """Sonraki sayfaya geç"""
        try:
            # Yöntem 1: DataTables next butonu
            next_selectors = [
                ".paginate_button.next:not(.disabled)",
                "#resultTable_next:not(.disabled)",
                ".dataTables_paginate .next:not(.disabled)",
                "a.paginate_button.next:not(.disabled)"
            ]
            
            for selector in next_selectors:
                btns = self.driver.find_elements(By.CSS_SELECTOR, selector)
                for btn in btns:
                    if btn.is_displayed() and btn.is_enabled():
                        self.driver.execute_script("arguments[0].click();", btn)
                        time.sleep(3)
                        return True
            
            # Yöntem 2: Sayfa numarasını artır
            page_inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='text'], input[type='number']")
            for inp in page_inputs:
                try:
                    val = inp.get_attribute("value")
                    if val and val.isdigit():
                        next_page = int(val) + 1
                        inp.clear()
                        inp.send_keys(str(next_page))
                        inp.send_keys(Keys.RETURN)
                        time.sleep(3)
                        return True
                except:
                    continue
            
            # Yöntem 3: › veya Sonraki linkine tıkla
            next_links = self.driver.find_elements(By.XPATH, 
                "//a[contains(text(),'Sonraki') or contains(text(),'›') or contains(text(),'>') or contains(text(),'İleri')]")
            for link in next_links:
                try:
                    parent_class = link.find_element(By.XPATH, "./..").get_attribute("class") or ""
                    if "disabled" not in parent_class and link.is_displayed():
                        self.driver.execute_script("arguments[0].click();", link)
                        time.sleep(3)
                        return True
                except:
                    continue
            
            print("   -> Sonraki sayfa butonu bulunamadı.")
            return False
            
        except Exception as e:
            print(f"   -> Sayfa geçiş hatası: {e}")
            return False

    def _save_json(self):
        """JSON dosyasına kaydet"""
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(self.cases, f, ensure_ascii=False, indent=2)
        print(f"   💾 {len(self.cases)} kayıt '{OUTPUT_FILE}' dosyasına kaydedildi.")

    def close(self):
        """Tarayıcıyı kapat"""
        self.driver.quit()


# Test fonksiyonu - Graph Links regex'lerini test et
def test_graph_links():
    """Regex pattern'larını örnek metin üzerinde test et"""
    sample_text = """
    Kiracı aleyhine düzenleme yasağı başlıklı 6098 Sayılı Türk Borçlar Kanununun 346.maddesinde; 
    kiracıya kira bedeli ve yan giderler dışında başka bir ödeme yükümlülüğü getirilemeyeceği, 
    özellikle kira bedelinin zamanında ödenmemesi halinde ceza koşulu ödeneceğine veya sonraki 
    kira bedellerinin muaccel olacağına ilişkin anlaşmaların geçersiz olduğu, 
    6101 Sayılı Türk Borçlar Kanununun Yürürlüğü ve Uygulama Şekli Hakkında Kanunun Geçmişe Etkili Olma 
    başlıklı 2.maddesinde; Türk Borçlar Kanununun kamu düzenine ve genel ahlaka ilişkin kurallarının 
    gerçekleştikleri tarihe bakılmaksızın bütün fiil ve işlemlere uygulanacağı, 
    aynı kanunun görülmekte olan davalara ilişkin uygulama başlıklı 7.maddesinde de; 
    Türk Borçlar Kanununun kamu düzenine ve genel ahlaka ilişkin kuralları ile geçici ödemelere ilişkin 
    76'ncı, faize ilişkin 88'nci, temerrüt faizine ilişkin 120'nci ve aşırı ifa güçlüğüne ilişkin 
    138'nci maddesinin görülmekte olan davalara da uygulanacağı hüküm altına alınmıştır.
    
    Bununla birlikte 6217 Sayılı Yasanın geçici 2.maddesinde değişiklik yapan 6353 Sayılı Yasanın 
    53.maddesine göre; kiracının Türk Ticaret Kanunun'da tacir olarak sayılan kişiler ile özel hukuk 
    ve kamu hukuku tüzel kişileri olduğu işyeri kiralarında 6098 Sayılı Türk Borçlar Kanununun 
    323, 325, 331, 340, 343, 344, 346 ve 354'ncü maddelerinin 1.7.2012 tarihinden itibaren 8 yıl 
    süreyle uygulanamayacağı
    
    6102 Sayılı T.T.K.nun 12.maddesine göre "bir ticari işletmeyi kısmen dahi olsa kendi adına 
    işleten kimseye tacir denir." Aynı yasanın 115.maddesi hükmünce
    
    5362 sayılı Esnaf ve Sanatkarlar Meslek Kuruluşları Kanunu'nun 3'üncü maddesinde
    
    Vergi Usul Kanunu'nun 177.maddesinin 1.fıkrasının 1 ve 3 nolu bentlerinde
    
    SONUÇ: 6100 sayılı HMK.ya 6217 Sayılı Kanunla eklenen geçici 3.madde hükmü gözetilerek 
    HUMK.nın 428 ve İİK.nın 366.maddesi uyarınca
    """
    
    print("=" * 60)
    print("GRAPH LINKS REGEX TEST")
    print("=" * 60)
    
    scraper = YargitayScraperV4.__new__(YargitayScraperV4)
    links = scraper._extract_all_law_references(sample_text)
    
    print(f"\n✅ Toplam {len(links)} referans bulundu:\n")
    for i, link in enumerate(links, 1):
        print(f"{i:2}. {link['target_id']:25} | {link['law_name'][:30]:30} | {link['text_span'][:40]}")
    
    return links


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test modu
        test_graph_links()
    else:
        # Normal scraping modu
        scraper = YargitayScraperV4()
        try:
            scraper.start()
        finally:
            scraper.close()
