import json
import re
import random
import time
import os
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from fetch_single_decision import get_decision_data
from dotenv import load_dotenv
from openai import OpenAI

# .env dosyasını yükle
load_dotenv()

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY .env dosyasında bulunamadı!")

# Law code to topic mapping
LAW_CODE_TO_TOPIC = {
    "5237": "Ceza Hukuku",      # TCK - Criminal
    "4721": "Aile ve Miras",    # TMK - Family/Civil
    "6102": "Ticaret Hukuku",   # TTK - Commercial
    "4857": "İş Hukuku",        # Labor Law
    "6098": "Borçlar Hukuku",   # TBK - Obligations
}

# Law names and abbreviations for masking
LAW_MASKING_PATTERNS = {
    "5237": [
        r"\b5237\b",
        r"\bTCK\b",
        r"Türk\s+Ceza\s+Kanunu",
        r"Türk\s+Ceza\s+Kanununun?",
        r"Türk\s+Ceza\s+Kanun[un]?",
    ],
    "4721": [
        r"\b4721\b",
        r"\bTMK\b",
        r"Türk\s+Medeni\s+Kanunu",
        r"Türk\s+Medeni\s+Kanununun?",
        r"Türk\s+Medeni\s+Kanun[un]?",
    ],
    "6102": [
        r"\b6102\b",
        r"\bTTK\b",
        r"Türk\s+Ticaret\s+Kanunu",
        r"Türk\s+Ticaret\s+Kanununun?",
        r"Türk\s+Ticaret\s+Kanun[un]?",
    ],
    "4857": [
        r"\b4857\b",
        r"İş\s+Kanunu",
        r"İş\s+Kanununun?",
        r"İş\s+Kanun[un]?",
    ],
    "6098": [
        r"\b6098\b",
        r"\bTBK\b",
        r"Türk\s+Borçlar\s+Kanunu",
        r"Türk\s+Borçlar\s+Kanununun?",
        r"Türk\s+Borçlar\s+Kanun[un]?",
    ],
}

# All law codes for comprehensive masking
ALL_LAW_CODES = list(LAW_CODE_TO_TOPIC.keys())


class SimilarDecisionScraper:
    """Similar decisions'ları scrape etmek için scraper"""
    
    def __init__(self, headless=False):
        self.base_url = "https://www.hukukturk.com/yargitay-kararlari"
        
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
    
    def parse_similar_decision(self, decision_str):
        """
        Parse similar decision string: "Yargıtay, 2. Hukuk Dairesi, E. 2017/3243, K. 2017/13068, T. 21.11.2017"
        Returns: {"esas_yil": "2017", "esas_no": "3243", "karar_yil": "2017", "karar_no": "13068"}
        """
        # Pattern: E. YYYY/XXXX, K. YYYY/XXXX
        match = re.search(r'E\.\s*(\d+)/(\d+).*?K\.\s*(\d+)/(\d+)', decision_str)
        if match:
            return {
                "esas_yil": match.group(1),
                "esas_no": match.group(2),
                "karar_yil": match.group(3),
                "karar_no": match.group(4)
            }
        return None
    
    def search_by_decision_numbers(self, esas_yil, esas_no, karar_yil, karar_no):
        """
        Esas No ve Karar No ile arama yap
        """
        try:
            # Siteye git
            self.driver.get(self.base_url)
            time.sleep(2)
            
            # Esas No alanlarını bul ve doldur
            esas_yil_input = self.driver.find_element(By.ID, "mainContent_ctl00_txtEsasNo1")
            esas_no_input = self.driver.find_element(By.ID, "mainContent_ctl00_txtEsasNo2")
            
            esas_yil_input.clear()
            esas_yil_input.send_keys(esas_yil)
            time.sleep(0.3)
            
            esas_no_input.clear()
            esas_no_input.send_keys(esas_no)
            time.sleep(0.3)
            
            # Karar No alanlarını bul ve doldur
            karar_yil_input = self.driver.find_element(By.ID, "mainContent_ctl00_txtKararNo1")
            karar_no_input = self.driver.find_element(By.ID, "mainContent_ctl00_txtKararNo2")
            
            karar_yil_input.clear()
            karar_yil_input.send_keys(karar_yil)
            time.sleep(0.3)
            
            karar_no_input.clear()
            karar_no_input.send_keys(karar_no)
            time.sleep(0.3)
            
            # Ara butonunu bul ve tıkla
            ara_button = None
            ara_selectors = [
                "//input[@value='ARA']",
                "//input[@value='Ara']",
                "//button[text()='ARA']",
                "//button[text()='Ara']",
                "//input[@type='submit' and contains(@value, 'ARA')]",
                "//input[@type='submit' and contains(@value, 'Ara')]",
            ]
            
            for selector in ara_selectors:
                try:
                    buttons = self.driver.find_elements(By.XPATH, selector)
                    for btn in buttons:
                        if btn.is_displayed() and btn.is_enabled():
                            ara_button = btn
                            break
                    if ara_button:
                        break
                except:
                    continue
            
            if ara_button:
                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", ara_button)
                time.sleep(0.3)
                self.driver.execute_script("arguments[0].click();", ara_button)
                time.sleep(3)
                
                # Sonuç sayfasındaki ilk linki bul
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                links = soup.find_all('a', href=lambda x: x and x.startswith('/Goster?v='))
                
                if links:
                    href = links[0].get('href', '')
                    return href
                
            return None
            
        except Exception as e:
            print(f"   ⚠️ Similar decision arama hatası: {e}")
            return None
    
    def scrape_similar_decision(self, decision_str):
        """
        Similar decision string'ini parse et ve scrape et
        Returns: decision_data dict veya None
        """
        parsed = self.parse_similar_decision(decision_str)
        if not parsed:
            return None
        
        href = self.search_by_decision_numbers(
            parsed["esas_yil"],
            parsed["esas_no"],
            parsed["karar_yil"],
            parsed["karar_no"]
        )
        
        if href:
            return get_decision_data(href)
        
        return None
    
    def close(self):
        """Tarayıcıyı kapat"""
        self.driver.quit()


def generate_query_from_case(full_text, client):
    """
    GPT-4o-mini ile karar metninden sadece olay/dava kısmını çıkarır.
    Kanun maddeleri, sonuçlar, karar numaraları vs. çıkarılmaz, sadece olay anlatılır.
    """
    system_prompt = """Sen uzman bir Türk Hukukçusun. Verilen Yargıtay karar metninden SADECE olayı/davayı anlat. 

ÖNEMLİ KURALLAR:
- Sadece olayın/davanın ne olduğunu anlat
- Kanun maddeleri, madde numaraları, sonuçlar, karar numaraları, tarihler gibi detayları ÇIKARMA
- Tarafların kim olduğunu, ne istediğini, olayın ne olduğunu anlat
- Hukuki terimleri kullan ama madde referansları verme
- Kısa ve öz olsun (maksimum 300 kelime)"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Yargıtay karar metni:\n\n{full_text}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"   ⚠️  LLM query generation hatası: {e}")
        # Fallback: full_text'in ilk 500 karakterini al
        return full_text[:500].strip()


def load_yargitay_dataset():
    """
    Load yargitay_dataset.json file.
    Returns: list of cases
    """
    dataset_file = Path("data/yargitay_dataset.json")
    
    if not dataset_file.exists():
        print(f"ERROR: {dataset_file} bulunamadı!")
        return []
    
    print(f"Loading {dataset_file}...")
    with open(dataset_file, "r", encoding="utf-8") as f:
        cases = json.load(f)
    
    print(f"  -> {len(cases)} cases loaded")
    return cases


def select_cases_by_law_code(cases, cases_per_law=10):
    """
    Her law_code için belirtilen sayıda case seç.
    Returns: list of selected cases (toplam cases_per_law * number_of_laws)
    """
    print(f"\n🔍 Her law_code için {cases_per_law} case seçiliyor...")
    
    # Cases'leri law_code'a göre grupla
    cases_by_law = {}
    for case in cases:
        law_code = case.get("law_code", "")
        if not law_code:
            continue
        
        if law_code not in cases_by_law:
            cases_by_law[law_code] = []
        cases_by_law[law_code].append(case)
    
    # Her law_code için cases_per_law kadar seç
    selected_cases = []
    for law_code, law_cases in cases_by_law.items():
        # Shuffle ve ilk cases_per_law kadarını seç
        shuffled = law_cases.copy()
        random.shuffle(shuffled)
        selected = shuffled[:cases_per_law]
        selected_cases.extend(selected)
        print(f"  -> {law_code}: {len(law_cases)} case arasından {len(selected)} seçildi")
    
    print(f"\n✅ Toplam {len(selected_cases)} case seçildi ({len(cases_by_law)} law_code x {cases_per_law} case)")
    return selected_cases


def process_case_and_create_test(case, idx, total, scraper, seen_similar_decisions, scraped_decisions_file, client, output_file):
    """
    Her case için: Similar decisions scrape et, sonra test case oluştur ve kaydet.
    """
    # Similar decisions'ları scrape et
    similar_decisions = case.get("similar_decisions", [])
    scraped_similar_decisions = []
    
    if similar_decisions:
        for sim_decision_str in similar_decisions[:5]:  # İlk 5 tanesini scrape et
            print(f"      🔍 Scraping: {sim_decision_str[:60]}...")
            scraped_data = scraper.scrape_similar_decision(sim_decision_str)
            
            if scraped_data and 'error' not in scraped_data:
                url = scraped_data.get("url", "")
                if url:
                    # Duplicate kontrolü - eğer daha önce scrape edildiyse cache'den al
                    if url in seen_similar_decisions:
                        scraped_data = seen_similar_decisions[url]
                        print(f"         ✅ Cache'den alındı")
                    else:
                        seen_similar_decisions[url] = scraped_data
                        print(f"         ✅ Başarılı")
                        # Incremental kaydet
                        save_similar_case_incremental(scraped_decisions_file, scraped_data)
                    scraped_similar_decisions.append(scraped_data)
            else:
                print(f"         ❌ Hata: {scraped_data.get('error', 'Bilinmeyen hata') if scraped_data else 'Veri alınamadı'}")
            
            time.sleep(1)  # Rate limiting
    
    # Case'i güncelle
    case_with_scraped = case.copy()
    case_with_scraped["scraped_similar_decisions"] = scraped_similar_decisions
    
    # Şimdi test case oluştur
    full_text = case_with_scraped.get("full_text", "")
    if not full_text or len(full_text) < 100:
        print(f"      ⏭️  Full text yok veya çok kısa, test case oluşturulamadı")
        return case_with_scraped, None
    
    relevant_legislation = case_with_scraped.get("relevant_legislation", [])
    ground_truth_relevant_legislation = relevant_legislation.copy()
    
    # Ground truth: similar_decisions (scraped olanların URL'lerini al)
    ground_truth_similar_cases = []
    for scraped in scraped_similar_decisions:
        url = scraped.get("url", "")
        if url:
            ground_truth_similar_cases.append(url)
    
    # Skip if no ground truth
    if not ground_truth_relevant_legislation and not ground_truth_similar_cases:
        print(f"      ⏭️  Ground truth yok, test case oluşturulamadı")
        return case_with_scraped, None
    
    # LLM ile query oluştur (sadece olay/dava kısmı)
    print(f"      🤖 Generating query with LLM...")
    query = generate_query_from_case(full_text, client)
    
    # Create test case
    test_case = {
        "query": query,
        "ground_truth_relevant_legislation": ground_truth_relevant_legislation,
        "ground_truth_similar_cases": ground_truth_similar_cases,
    }
    
    # Incremental saving - test case'i hemen kaydet
    save_benchmark_incremental(output_file, test_case, total)
    
    time.sleep(0.5)  # Rate limiting for LLM calls
    
    return case_with_scraped, test_case


def scrape_similar_decisions_for_cases(cases, client, output_file):
    """
    Tüm cases için similar decisions'ları scrape et ve test case'leri oluştur.
    Her case işlendikten sonra hem similar case kaydedilir hem de test case oluşturulup kaydedilir.
    Returns: tuple (processed_cases, test_cases)
    """
    print("\n" + "="*60)
    print("SIMILAR DECISIONS SCRAPER + TEST CASE CREATOR")
    print("="*60)
    
    print(f"\nToplam {len(cases)} case işlenecek")
    print(f"Her case için: Similar decisions scrape edilecek, sonra test case oluşturulup kaydedilecek\n")
    
    # Scraped decisions dosyası
    scraped_decisions_file = Path("data/similar_cases.json")
    
    # Scraper oluştur
    scraper = SimilarDecisionScraper(headless=False)
    
    processed_cases = []
    test_cases = []
    seen_similar_decisions = {}  # URL bazında duplicate kontrolü (memory cache)
    
    # Mevcut similar cases'leri yükle (eğer varsa)
    if scraped_decisions_file.exists():
        with open(scraped_decisions_file, "r", encoding="utf-8") as f:
            existing_cases = json.load(f)
            for existing_case in existing_cases:
                url = existing_case.get("url", "")
                if url:
                    seen_similar_decisions[url] = existing_case
        print(f"  📂 {len(seen_similar_decisions)} mevcut similar case yüklendi\n")
    
    try:
        for idx, case in enumerate(cases, 1):
            print(f"\n  [{idx}/{len(cases)}] Processing case: {case.get('decision_number', 'N/A')}")
            
            # Similar decisions varsa log yaz
            similar_decisions = case.get("similar_decisions", [])
            if not similar_decisions:
                print(f"      ⏭️  Similar decisions yok, test case oluşturulacak...")
            
            # Her case için: scrape et ve test case oluştur
            case_with_scraped, test_case = process_case_and_create_test(
                case, idx, len(cases), scraper, seen_similar_decisions, 
                scraped_decisions_file, client, output_file
            )
            
            processed_cases.append(case_with_scraped)
            if test_case:
                test_cases.append(test_case)
    
    finally:
        scraper.close()
    
    print(f"\n📊 Toplam {len(seen_similar_decisions)} unique similar decision scrape edildi")
    print(f"  ✅ Tüm similar cases '{scraped_decisions_file}' dosyasına kaydedildi")
    print(f"📊 Toplam {len(test_cases)} test case oluşturuldu ve kaydedildi")
    
    return processed_cases, test_cases


def save_similar_case_incremental(scraped_decisions_file, scraped_data):
    """
    Her similar case scrape edildiğinde JSON dosyasına incremental olarak kaydet.
    Not: Bu fonksiyon çağrılmadan önce duplicate kontrolü yapılmış olmalı.
    """
    # Eğer dosya yoksa, yeni bir liste oluştur
    if not scraped_decisions_file.exists():
        scraped_decisions_list = []
    else:
        # Mevcut dosyayı yükle
        with open(scraped_decisions_file, "r", encoding="utf-8") as f:
            scraped_decisions_list = json.load(f)
    
    # Yeni case'i ekle (duplicate kontrolü zaten yapılmış)
    scraped_decisions_list.append(scraped_data)
    
    # Dosyaya kaydet
    with open(scraped_decisions_file, "w", encoding="utf-8") as f:
        json.dump(scraped_decisions_list, f, ensure_ascii=False, indent=2)
    
    print(f"         💾 Similar case kaydedildi (toplam: {len(scraped_decisions_list)})")


def load_scraped_similar_decisions():
    """
    Similar cases JSON dosyasını yükle.
    Returns: list of scraped decisions
    """
    scraped_decisions_file = Path("data/similar_cases.json")
    
    if not scraped_decisions_file.exists():
        print(f"⚠️  {scraped_decisions_file} bulunamadı, boş liste döndürülüyor...")
        return []
    
    print(f"📂 Loading similar cases from {scraped_decisions_file}...")
    with open(scraped_decisions_file, "r", encoding="utf-8") as f:
        scraped_decisions = json.load(f)
    
    print(f"  -> {len(scraped_decisions)} similar case yüklendi")
    return scraped_decisions


def save_benchmark_incremental(output_file, test_case, total_count):
    """
    Her test case'i JSON dosyasına incremental olarak kaydet.
    """
    # Eğer dosya yoksa, yeni bir benchmark oluştur
    if not os.path.exists(output_file):
        benchmark = {
            "version": "3.0",
            "total_cases": 0,
            "query_generation": "gpt-4o-mini (olay/dava extraction only)",
            "test_cases": []
        }
    else:
        # Mevcut dosyayı yükle
        with open(output_file, "r", encoding="utf-8") as f:
            benchmark = json.load(f)
    
    # Yeni test case'i ekle
    benchmark["test_cases"].append(test_case)
    benchmark["total_cases"] = len(benchmark["test_cases"])
    
    # Dosyaya kaydet
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, ensure_ascii=False, indent=2)
    
    print(f"      💾 [{benchmark['total_cases']}/{total_count}] Test case kaydedildi")


def create_test_cases(cases, client, output_file):
    """
    Create test cases from scraped cases.
    Her test case oluşturulduğunda JSON'a kaydedilir (incremental saving).
    Returns: list of test case dictionaries.
    Each test case contains: query, ground_truth_relevant_legislation, ground_truth_similar_cases
    """
    all_test_cases = []
    
    print(f"\nCreating test cases for {len(cases)} cases...")
    print(f"Her test case tamamlanınca '{output_file}' dosyasına kaydedilecek\n")
    
    for idx, case in enumerate(cases, 1):
        # Extract required fields
        full_text = case.get("full_text", "")
        relevant_legislation = case.get("relevant_legislation", [])
        scraped_similar_decisions = case.get("scraped_similar_decisions", [])
        law_code = case.get("law_code", "")
        
        # Skip if essential fields are missing
        if not full_text or len(full_text) < 100:
            print(f"  [{idx}/{len(cases)}] ⏭️  Full text yok veya çok kısa, atlanıyor...")
            continue
        
        # Ground truth: relevant_legislation
        ground_truth_relevant_legislation = relevant_legislation.copy()
        
        # Ground truth: similar_decisions (scraped olanların URL'lerini al)
        ground_truth_similar_cases = []
        for scraped in scraped_similar_decisions:
            url = scraped.get("url", "")
            if url:
                ground_truth_similar_cases.append(url)
        
        # Skip if no ground truth
        if not ground_truth_relevant_legislation and not ground_truth_similar_cases:
            print(f"  [{idx}/{len(cases)}] ⏭️  Ground truth yok, atlanıyor...")
            continue
        
        # LLM ile query oluştur (sadece olay/dava kısmı)
        print(f"  [{idx}/{len(cases)}] Generating query with LLM...")
        query = generate_query_from_case(full_text, client)
        
        # Create test case - sadece query, ground_truth_relevant_legislation, ground_truth_similar_cases
        test_case = {
            "query": query,
            "ground_truth_relevant_legislation": ground_truth_relevant_legislation,
            "ground_truth_similar_cases": ground_truth_similar_cases,
        }
        
        all_test_cases.append(test_case)
        
        # Incremental saving - her test case'i hemen kaydet
        save_benchmark_incremental(output_file, test_case, len(cases))
        
        # Rate limiting for LLM calls
        time.sleep(0.5)
    
    print(f"\n✅ Created {len(all_test_cases)} valid test cases")
    
    return all_test_cases


def main():
    print("=" * 60)
    print("MULTI-LAW BENCHMARK GENERATOR")
    print("=" * 60)
    
    # OpenAI client oluştur
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Output file
    output_file = "data/multi_law_benchmark.json"
    
    # Load yargitay dataset
    print("\n[1/4] Loading yargitay dataset...")
    all_cases = load_yargitay_dataset()
    
    if not all_cases:
        print("ERROR: No cases found! Please run the scraper first.")
        return
    
    # Her law_code için 10 case seç (toplam 50 case)
    print("\n[2/3] Selecting cases (10 per law_code, total 50)...")
    selected_cases = select_cases_by_law_code(all_cases, cases_per_law=10)
    
    if not selected_cases:
        print("ERROR: No cases selected!")
        return
    
    # Scrape similar decisions ve test case oluştur (her case için hemen işlenir)
    print(f"\n[3/3] Processing cases: Similar decisions scraping + Test case creation...")
    print(f"Output file: {output_file}")
    cases_with_scraped, test_cases = scrape_similar_decisions_for_cases(selected_cases, client, output_file)
    
    if not cases_with_scraped:
        print("ERROR: No cases processed!")
        return
    
    print(f"\n✅ Total test cases created: {len(test_cases)}")
    
    # Show ground truth statistics
    print("\nGround truth statistics:")
    relevant_legislation_count = sum(1 for tc in test_cases if tc.get("ground_truth_relevant_legislation"))
    similar_cases_count = sum(1 for tc in test_cases if tc.get("ground_truth_similar_cases"))
    print(f"  Cases with relevant_legislation ground truth: {relevant_legislation_count}")
    print(f"  Cases with similar_cases ground truth: {similar_cases_count}")
    
    # Final summary (dosya zaten kaydedildi, sadece özet göster)
    print(f"\n✅ Benchmark generation completed!")
    print(f"\nBenchmark Summary:")
    print(f"  Total test cases: {len(test_cases)} (10 per law_code x 5 laws)")
    print(f"  Query generation: GPT-4o-mini (olay/dava only)")
    print(f"  Output file: {output_file}")
    print(f"  Similar cases: data/similar_cases.json")
    print("=" * 60)


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()
