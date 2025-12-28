import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from fetch_single_decision import get_decision_data

# --- YAPILANDIRMA ---
TARGET_COUNT = 100  # Her kanun için 100 karar hedefi

# Target laws to scrape
TARGET_LAWS = [
    "4721",  # TMK - Türk Medeni Kanunu (Civil Law / Divorce, Inheritance)
    "Türk Ceza Kanunu",  # TCK - Türk Ceza Kanunu (Penal Law / Murder, Theft)
    "6102",  # TTK - Türk Ticaret Kanunu (Commercial Law)
    "4857",  # İş Kanunu (Labor Law)
    "6098",  # TBK - Türk Borçlar Kanunu (Obligations - Existing)
]

# Kanun adları ve kısaltmaları için sözlük
LAW_NAMES = {
    "6098": "Türk Borçlar Kanunu (TBK)",
    "6102": "Türk Ticaret Kanunu (TTK)",
    "4721": "Türk Medeni Kanunu (TMK)",
    "5237": "Türk Ceza Kanunu (TCK)",
    "4857": "İş Kanunu (Labor Law)",
}


class HukukturkScraper:
    def __init__(self, law_code="6098", headless=False):
        self.law_code = law_code
        self.output_file = "data/yargitay_dataset.json"  # Tüm kanunlar için tek dosya
        self.base_url = "https://www.hukukturk.com/Default.aspx?pageid=5&Mevzuat=31696"
        
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        self.driver.implicitly_wait(10)
        self.wait = WebDriverWait(self.driver, 20)
        self.all_cases = []  # Tüm kanunların verileri burada
        self.collected_count = 0
        self.seen_urls = set()  # Duplikat kontrolü için

    def start(self):
        print("=" * 60)
        print("HUKUKTURK.COM KARAR SCRAPER")
        print(f"KANUN KODU: {self.law_code} ({LAW_NAMES.get(self.law_code, 'Bilinmeyen')})")
        print(f"ÇIKTI DOSYASI: {self.output_file}")
        print("=" * 60)
        
        # Mevcut verileri yükle (eğer varsa)
        self._load_existing_data()
        
        # Siteye git
        print(f"\n🌐 {self.base_url} adresine gidiliyor...")
        self.driver.get(self.base_url)
        time.sleep(3)
        
        # Kanun field'ına yaz ve autocomplete'ten seç
        self._search_by_law_autocomplete()
        
        # Sonuçları topla
        self._scrape_results()
        
        # Sonuçları kaydet
        self._save_json()
        
        print(f"\n{'='*60}")
        print(f"🎉 TAMAMLANDI! Bu kanun için {self.collected_count} karar toplandı!")
        print(f"📊 Toplam dataset: {len(self.all_cases)} karar")
        print(f"{'='*60}")

    def _search_by_law_autocomplete(self):
        """Kanun field'ına kanun numarasını yaz, autocomplete'ten seç ve ara"""
        print(f"\n🔍 Kanun field'ına '{self.law_code}' yazılıyor...")
        
        try:
            # Kanun input field'ını bul (farklı selector'ları dene)
            kanun_input = None
            selectors = [
                "//input[@placeholder='İlgili mevzuatın numarasını veya adını girin...']",
                "//input[@placeholder='İlgili mevzuatın numarasını veya adını girin']",
                "//input[contains(@placeholder, 'mevzuat')]",
                "//input[contains(@placeholder, 'Mevzuat')]",
                "//input[@id='ctl00_ContentPlaceHolder1_txtKanun']",
                "//input[@name*='Kanun' or @name*='kanun']",
                "//input[@type='text' and contains(@id, 'Kanun')]",
            ]
            
            for selector in selectors:
                try:
                    kanun_input = self.driver.find_element(By.XPATH, selector)
                    if kanun_input and kanun_input.is_displayed():
                        print(f"   ✅ Kanun input bulundu: {selector}")
                        break
                except:
                    continue
            
            if not kanun_input:
                print("   ❌ Kanun input field bulunamadı!")
                print("   Sayfanın HTML'ini kontrol ediyorum...")
                input("Lütfen sayfayı kontrol edin ve ENTER'a basın...")
                return
            
            # Field'a scroll et
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", kanun_input)
            time.sleep(0.5)
            
            # Field'ı temizle
            kanun_input.clear()
            time.sleep(0.3)
            
            # Kanun numarasını yaz
            print(f"   📝 '{self.law_code}' yazılıyor...")
            kanun_input.send_keys(self.law_code)
            time.sleep(2)  # Autocomplete'in görünmesini bekle
            
            # Sayfa kaydırma ile autocomplete görünür hale getir
            self.driver.execute_script("window.scrollBy(0, 100);")
            time.sleep(1)
            
            # Autocomplete dropdown'ından ilk öneriyi seç
            try:
                print("   🔍 Autocomplete listesi aranıyor...")
                # Farklı autocomplete selector'ları dene
                autocomplete_selectors = [
                    "ul.ui-autocomplete li",
                    ".ui-autocomplete li",
                    "[role='listbox'] li",
                    "[role='option']",
                    ".autocomplete-item",
                    "#ui-id-1 li",
                    "ul[id*='ui-id'] li",
                    "li.ui-menu-item",
                ]
                
                autocomplete_items = None
                for selector in autocomplete_selectors:
                    try:
                        items = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        visible_items = [item for item in items if item.is_displayed()]
                        if visible_items:
                            autocomplete_items = visible_items
                            print(f"   ✅ Autocomplete bulundu: {selector} ({len(visible_items)} öğe)")
                            break
                    except:
                        continue
                
                if autocomplete_items and len(autocomplete_items) > 0:
                    first_item_text = autocomplete_items[0].text[:80]
                    print(f"   ✅ İlk öneri: {first_item_text}...")
                    print(f"   👆 İlk öneriye tıklanıyor...")
                    self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", autocomplete_items[0])
                    time.sleep(0.3)
                    autocomplete_items[0].click()
                    time.sleep(1.5)
                else:
                    print("   ⚠️ Autocomplete öğeleri görünür değil, Enter tuşuna basılıyor...")
                    kanun_input.send_keys(Keys.RETURN)
                    time.sleep(2)
            except Exception as e:
                print(f"   ⚠️ Autocomplete seçilemedi: {e}")
                print("   Enter tuşuna basılıyor...")
                kanun_input.send_keys(Keys.RETURN)
                time.sleep(2)
            
            # Ara butonunu bul ve tıkla
            print("   🔍 Ara butonu aranıyor...")
            ara_button = None
            ara_selectors = [
                "//input[@value='ARA']",
                "//input[@value='Ara']",
                "//button[text()='ARA']",
                "//button[text()='Ara']",
                "//input[@type='submit' and contains(@value, 'ARA')]",
                "//input[@type='submit' and contains(@value, 'Ara')]",
                "//input[@id*='btnAra' or @id*='btnSearch']",
                "//button[@id*='btnAra' or @id*='btnSearch']",
            ]
            
            for selector in ara_selectors:
                try:
                    buttons = self.driver.find_elements(By.XPATH, selector)
                    for btn in buttons:
                        if btn.is_displayed() and btn.is_enabled():
                            ara_button = btn
                            print(f"   ✅ Ara butonu bulundu: {selector}")
                            break
                    if ara_button:
                        break
                except:
                    continue
            
            if not ara_button:
                print("   ❌ Ara butonu bulunamadı!")
                input("Lütfen manuel olarak Ara butonuna tıklayın ve ENTER'a basın...")
            else:
                print("   👆 Ara butonuna tıklanıyor...")
                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", ara_button)
                time.sleep(0.3)
                self.driver.execute_script("arguments[0].click();", ara_button)
                time.sleep(3)  # Sonuçların yüklenmesini bekle
            
            # Sonuç sayısını kontrol et
            try:
                result_text = self.driver.find_element(By.XPATH, "//*[contains(text(), 'kayıt bulundu') or contains(text(), 'üzerinde kayıt')]").text
                print(f"   ✅ {result_text}")
            except:
                print("   ✅ Arama tamamlandı, sonuçlar yükleniyor...")
            
        except Exception as e:
            print(f"❌ Arama işlemi hatası: {e}")
            import traceback
            traceback.print_exc()
            input("Hata oluştu, devam etmek için ENTER'a basın...")

    def _scrape_results(self):
        """Sonuçlardan href'leri topla ve fetch_single_decision.py ile işle (Sayfalama desteği ile)"""
        print(f"\n🔍 Sonuçlardan href'ler toplanıyor (Hedef: {TARGET_COUNT} karar)...")
        
        page_number = 1
        
        # Sayfalama döngüsü - 100 tane item toplanana kadar devam et
        while self.collected_count < TARGET_COUNT:
            print(f"\n📄 Sayfa {page_number} işleniyor...")
            
            # Sayfanın yüklenmesini bekle
            time.sleep(2)
            
            # Sonuç sayfasındaki tüm linkleri bul
            hrefs = self._extract_result_hrefs()
            
            if not hrefs:
                print("⚠️ Bu sayfada hiç href bulunamadı!")
                # Sonraki sayfaya geçmeyi dene
                if not self._go_to_next_page():
                    print("⚠️ Sonraki sayfa bulunamadı, scraping sonlandırılıyor.")
                    break
                page_number += 1
                continue
            
            print(f"✅ Sayfa {page_number}: {len(hrefs)} href bulundu")
            
            # Bu sayfadaki href'leri işle
            processed_this_page = 0
            for idx, href in enumerate(hrefs, 1):
                if self.collected_count >= TARGET_COUNT:
                    break
                
                # Duplikat kontrolü
                if href in self.seen_urls:
                    print(f"   [{self.collected_count + 1}/{TARGET_COUNT}] ⏭️  Zaten işlenmiş: {href}")
                    continue
                
                try:
                    print(f"   [{self.collected_count + 1}/{TARGET_COUNT}] 📥 İşleniyor: {href}")
                    
                    # fetch_single_decision.py'yi kullan
                    decision_data = get_decision_data(href)
                    
                    if decision_data and 'error' not in decision_data:
                        # Veriyi dataset formatına dönüştür
                        processed_data = self._convert_to_dataset_format(decision_data)
                        
                        if processed_data:
                            self.all_cases.append(processed_data)
                            self.seen_urls.add(href)
                            self.collected_count += 1
                            processed_this_page += 1
                            print(f"      ✅ Başarılı: {decision_data.get('chamber', 'N/A')} - {decision_data.get('decision_number', 'N/A')}")
                        else:
                            print(f"      ⚠️ Veri işlenemedi")
                    else:
                        error_msg = decision_data.get('error', 'Bilinmeyen hata') if decision_data else 'Veri alınamadı'
                        print(f"      ❌ Hata: {error_msg}")
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"      ❌ Hata: {str(e)[:80]}")
                    continue
            
            print(f"   📊 Bu sayfada {processed_this_page} yeni karar işlendi. Toplam: {self.collected_count}/{TARGET_COUNT}")
            
            # Hedef sayıya ulaşıldı mı kontrol et
            if self.collected_count >= TARGET_COUNT:
                print(f"\n✅ Hedef sayıya ulaşıldı: {self.collected_count}/{TARGET_COUNT}")
                break
            
            # Sonraki sayfaya geç
            if not self._go_to_next_page():
                print("⚠️ Sonraki sayfa bulunamadı, scraping sonlandırılıyor.")
                break
            
            page_number += 1
        
        print(f"\n✅ Bu kanun için toplam {self.collected_count} karar toplandı.")

    def _extract_result_hrefs(self):
        """Sonuç sayfasındaki karar linklerinin href'lerini topla"""
        hrefs = []
        
        try:
            # Sayfa kaynağını al
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Örnek: <a id="mainContent_ctl00_repeater_ctl00_0_lnkBaslik_0" class="detay" href="/Goster?v=GE95G9i5Huq">
            # Tüm /Goster?v= ile başlayan linkleri bul
            links = soup.find_all('a', href=lambda x: x and x.startswith('/Goster?v='))
            
            for link in links:
                href = link.get('href', '')
                if href and href not in hrefs:
                    hrefs.append(href)
            
            # Alternatif: Selenium ile direkt bul
            if not hrefs:
                elements = self.driver.find_elements(By.XPATH, "//a[starts-with(@href, '/Goster?v=')]")
                for elem in elements:
                    href = elem.get_attribute('href')
                    if href:
                        # Tam URL ise sadece path'i al
                        if href.startswith('http'):
                            from urllib.parse import urlparse
                            href = urlparse(href).path
                        if href not in hrefs:
                            hrefs.append(href)
            
        except Exception as e:
            print(f"   ⚠️ Href çıkarma hatası: {e}")
        
        return hrefs
    
    def _go_to_next_page(self):
        """Sonraki sayfaya geç (pagination)"""
        try:
            # Sayfa kaynağını al
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Sonraki sayfa linkini bul - örnek: <a href="/yargitay-kararlari?Mevzuat=31696&amp;p=3" title="Sayfaya Gitmek için İleri 3">
            # Farklı selector'ları dene
            next_page_link = None
            
            # Önce BeautifulSoup ile bul
            next_links = soup.find_all('a', href=lambda x: x and 'p=' in str(x) and ('İleri' in str(x.get('title', '')) or 'ileri' in str(x.get('title', '').lower())))
            
            # Alternatif: title'da "İleri" veya "ileri" geçen linkler
            if not next_links:
                next_links = soup.find_all('a', title=lambda x: x and ('İleri' in str(x) or 'ileri' in str(x).lower()))
            
            # Alternatif: fa-angle-right icon'u olan linkler
            if not next_links:
                next_links = soup.find_all('a', href=lambda x: x and 'p=' in str(x))
                # İçinde fa-angle-right span'ı olanları filtrele
                next_links = [link for link in next_links if link.find('span', class_=lambda x: x and 'fa-angle-right' in str(x))]
            
            if next_links:
                # En son sayfa numarasına sahip olanı bul (en büyük p= değeri)
                max_page = 0
                for link in next_links:
                    href = link.get('href', '')
                    # p= değerini çıkar
                    import re
                    match = re.search(r'p=(\d+)', str(href))
                    if match:
                        page_num = int(match.group(1))
                        if page_num > max_page:
                            max_page = page_num
                            next_page_link = link
            
            # Eğer BeautifulSoup ile bulamadıysak Selenium ile dene
            if not next_page_link:
                # Selenium ile sonraki sayfa linkini bul
                next_selectors = [
                    "//a[contains(@title, 'İleri')]",
                    "//a[contains(@title, 'ileri')]",
                    "//a[contains(@href, 'p=') and .//span[contains(@class, 'fa-angle-right')]]",
                    "//a[contains(@href, 'p=') and contains(@title, 'Sayfaya Gitmek')]",
                ]
                
                for selector in next_selectors:
                    try:
                        elements = self.driver.find_elements(By.XPATH, selector)
                        if elements:
                            # En son sayfa numarasına sahip olanı bul
                            max_page = 0
                            selected_elem = None
                            for elem in elements:
                                href = elem.get_attribute('href')
                                if href:
                                    import re
                                    match = re.search(r'p=(\d+)', href)
                                    if match:
                                        page_num = int(match.group(1))
                                        if page_num > max_page:
                                            max_page = page_num
                                            selected_elem = elem
                            
                            if selected_elem:
                                next_page_link = selected_elem
                                break
                    except:
                        continue
            
            if next_page_link:
                # Eğer BeautifulSoup elementi ise Selenium elementine çevir
                if hasattr(next_page_link, 'get'):
                    # BeautifulSoup elementi - href'ini al ve Selenium ile bul
                    href = next_page_link.get('href', '')
                    if href:
                        # Relative URL ise tam URL'ye çevir
                        if not href.startswith('http'):
                            from urllib.parse import urljoin
                            href = urljoin(self.driver.current_url, href)
                        
                        # Linke tıkla
                        print(f"   ➡️  Sonraki sayfaya geçiliyor: {href}")
                        self.driver.get(href)
                        time.sleep(3)  # Sayfanın yüklenmesini bekle
                        return True
                else:
                    # Selenium elementi - direkt tıkla
                    print(f"   ➡️  Sonraki sayfaya geçiliyor...")
                    self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_page_link)
                    time.sleep(0.5)
                    self.driver.execute_script("arguments[0].click();", next_page_link)
                    time.sleep(3)  # Sayfanın yüklenmesini bekle
                    return True
            else:
                print("   ⚠️ Sonraki sayfa linki bulunamadı.")
                return False
                
        except Exception as e:
            print(f"   ⚠️ Sonraki sayfaya geçiş hatası: {e}")
            return False
    
    def _convert_to_dataset_format(self, decision_data):
        """fetch_single_decision.py'den gelen veriyi dataset formatına dönüştür"""
        try:
            # Basit bir format - gerekirse daha detaylı yapılabilir
            return {
                "url": decision_data.get('url', ''),
                "chamber": decision_data.get('chamber', ''),
                "case_number": decision_data.get('case_number', ''),
                "decision_number": decision_data.get('decision_number', ''),
                "decision_date": decision_data.get('decision_date', ''),
                "summary": decision_data.get('summary', ''),
                "relevant_legislation": decision_data.get('relevant_legislation', []),
                "similar_decisions": decision_data.get('similar_decisions', []),
                "full_text": decision_data.get('full_text', ''),
                "conclusion": decision_data.get('conclusion', ''),
                "law_code": self.law_code  # Hangi kanun için toplandığını belirt
            }
        except Exception as e:
            print(f"      ⚠️ Format dönüşüm hatası: {e}")
            return None
    
    def _load_existing_data(self):
        """Mevcut dataset'i yükle (varsa)"""
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                self.all_cases = json.load(f)
                # Mevcut URL'leri seen_urls'e ekle
                for case in self.all_cases:
                    url = case.get('url', '')
                    if url:
                        # URL'den path'i çıkar
                        if url.startswith('http'):
                            from urllib.parse import urlparse
                            url = urlparse(url).path
                        self.seen_urls.add(url)
                print(f"📂 Mevcut dataset yüklendi: {len(self.all_cases)} karar")
        except FileNotFoundError:
            print("📂 Yeni dataset oluşturuluyor...")
            self.all_cases = []
        except Exception as e:
            print(f"⚠️ Dataset yükleme hatası: {e}")
            self.all_cases = []
    
    def _save_json(self):
        """JSON dosyasına kaydet"""
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(self.all_cases, f, ensure_ascii=False, indent=2)
        print(f"   💾 {len(self.all_cases)} kayıt '{self.output_file}' dosyasına kaydedildi.")

    def close(self):
        """Tarayıcıyı kapat"""
        self.driver.quit()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        # Single law code specified via command line
        law_code = sys.argv[1]
        print(f"Tek kanun modu: {law_code}")
        scraper = HukukturkScraper(law_code=law_code, headless=False)
        try:
            scraper.start()
        finally:
            scraper.close()
    else:
        # Multi-law mode: Loop through all target laws
        print(f"Çoklu kanun modu: {len(TARGET_LAWS)} kanun için scraping başlatılıyor...")
        for idx, law_code in enumerate(TARGET_LAWS, 1):
            print("\n" + "="*60)
            print(f"[{idx}/{len(TARGET_LAWS)}] KANUN: {law_code} ({LAW_NAMES.get(law_code, 'Bilinmeyen')})")
            print("="*60)
            
            scraper = HukukturkScraper(law_code=law_code, headless=False)
            try:
                scraper.start()
                print(f"\n✅ {law_code} için scraping tamamlandı. Toplam {scraper.collected_count} karar toplandı.")
            except KeyboardInterrupt:
                print(f"\n⚠️  {law_code} için scraping kullanıcı tarafından durduruldu.")
                scraper.close()
                break
            except Exception as e:
                print(f"\n❌ {law_code} için scraping hatası: {e}")
                import traceback
                traceback.print_exc()
            finally:
                scraper.close()
            
            if idx < len(TARGET_LAWS):
                print(f"\n⏳ Sonraki kanuna geçiliyor... (3 saniye)")
                time.sleep(3)
        
        print("\n" + "="*60)
        print("🎉 TÜM KANUNLAR İÇİN SCRAPING TAMAMLANDI!")
        print("="*60)
