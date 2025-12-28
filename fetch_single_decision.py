import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin

# Sabit Domain
BASE_URL = "https://www.hukukturk.com"

def get_decision_data(input_path):
    """
    Verilen URL path'ini (örn: /Goster?v=...) alır,
    HukukTurk sitesinden veriyi çeker ve sözlük (dict) olarak döndürür.
    """
    
    # Eğer input tam link değilse (http ile başlamıyorsa) domain ile birleştir
    if not input_path.startswith("http"):
        full_url = urljoin(BASE_URL, input_path)
    else:
        full_url = input_path

    # Request Ayarları
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7'
    }

    try:
        response = requests.get(full_url, headers=headers, timeout=15)
        response.raise_for_status()
        response.encoding = response.apparent_encoding

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Data Objesi (İngilizce Keyler)
        data = {
            "url": full_url,
            "chamber": None,               # Daire
            "case_number": None,           # Esas No
            "decision_number": None,       # Karar No
            "decision_date": None,         # Karar Tarihi
            "summary": None,               # Özü
            "relevant_legislation": [],    # İlgili Mevzuat
            "similar_decisions": [],       # Benzer Kararlar
            "full_text": None,             # Tam Metin
            "conclusion": None             # Sonuç
        }

        # --- Veri Çekme İşlemleri ---
        
        # Daire
        daire_div = soup.find('div', id='phMain_pnlMerci')
        if daire_div and daire_div.find('h4'):
            data['chamber'] = daire_div.find('h4').get_text(strip=True)

        # Helper
        def get_text_by_id(elem_id):
            el = soup.find(id=elem_id)
            return el.get_text(strip=True) if el else None

        data['case_number'] = get_text_by_id('phMain_lblKararEsasNo')
        data['decision_number'] = get_text_by_id('phMain_lblKararKararNo')
        data['decision_date'] = get_text_by_id('phMain_lblKararKararTarihi')
        data['summary'] = get_text_by_id('phMain_lblKararOzu')

        # Mevzuat - Statute encoding formatına çevir (LAW:kanun_no-madde_no)
        mevzuat_panel = soup.find('div', id='phMain_pnlMevzuatMadde')
        if mevzuat_panel:
            container = mevzuat_panel.find('div', class_='col-sm-10')
            if container and container.find('ul', recursive=False):
                for kanun_li in container.find('ul', recursive=False).find_all('li', recursive=False):
                    kanun_link = kanun_li.find('a', recursive=False)
                    if kanun_link:
                        kanun_adi = kanun_link.get_text(" ", strip=True)
                        sub_ul = kanun_li.find('ul')
                        maddeler = [m.get_text(strip=True) for m in sub_ul.find_all('a')] if sub_ul else []
                        
                        # Kanun numarasını çıkar (örn: "6098       TÜRK BORÇLAR KANUNU" -> "6098")
                        # İlk kelime genellikle kanun numarasıdır
                        kanun_no = None
                        kanun_adi_parts = kanun_adi.split()
                        if kanun_adi_parts:
                            # İlk kısım sayı ise kanun numarasıdır
                            first_part = kanun_adi_parts[0].strip()
                            if first_part.isdigit():
                                kanun_no = first_part
                        
                        # Eğer kanun numarası bulunamazsa, kanun_adi'den regex ile çıkar
                        if not kanun_no:
                            # Örnek: "6098       TÜRK BORÇLAR KANUNU" formatından sayıyı çıkar
                            match = re.search(r'^(\d+)', kanun_adi.strip())
                            if match:
                                kanun_no = match.group(1)
                        
                        # Madde numaralarını parse et
                        if kanun_no:
                            if maddeler:
                                # Her madde için LAW:kanun_no-madde_no formatında ID oluştur
                                for madde in maddeler:
                                    # Madde numarasını çıkar (örn: "Madde 49" -> "49")
                                    madde_match = re.search(r'(\d+)', madde)
                                    if madde_match:
                                        madde_no = madde_match.group(1)
                                        statute_id = f"LAW:{kanun_no}-{madde_no}"
                                        data['relevant_legislation'].append(statute_id)
                            else:
                                # Madde belirtilmemişse, sadece kanun numarası ile oluştur
                                statute_id = f"LAW:{kanun_no}"
                                data['relevant_legislation'].append(statute_id)

        # Benzer Kararlar
        benzer_panel = soup.find('div', id='phMain_pnlKararBenzer')
        if benzer_panel:
            for link in benzer_panel.find_all('a', id=re.compile('phMain_rptKararBenzer')):
                data['similar_decisions'].append(link.get_text(strip=True))

        # Tam Metin ve Sonuç
        text_div = soup.find('div', id='text')
        if text_div:
            paragraphs = [p.get_text(strip=True) for p in text_div.find_all('p')]
            full_text_str = "\n\n".join(paragraphs)
            data['full_text'] = full_text_str

            # Sonuç Kısmını Parsela (Regex)
            # Hem "SONUÇ :" hem "SONUÇ:" ihtimallerini yakalar, satır sonuna kadar alır.
            match = re.search(r'(SONUÇ\s*[:].*)', full_text_str, re.DOTALL | re.IGNORECASE)
            
            if match:
                data['conclusion'] = match.group(1).strip()
            else:
                # Fallback: Eğer regex bulamazsa son paragrafa bak
                if paragraphs and paragraphs[-1].strip().upper().startswith("SONUÇ"):
                    data['conclusion'] = paragraphs[-1].strip()

        return data

    except Exception as e:
        # Hata durumunda boş veri ve hata mesajı dön
        return {"url": full_url, "error": str(e)}

# Bu blok sadece bu dosya doğrudan çalıştırılırsa çalışır (Test için)
if __name__ == "__main__":
    test_input = "/Goster?v=GE95G9i5Huq"
    print(get_decision_data(test_input))