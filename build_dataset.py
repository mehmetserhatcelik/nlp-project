import requests
from bs4 import BeautifulSoup
import re
import json
import time

# --- CONFIGURATION ---
LAW_NO = "6098"
LAW_URL = "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.6098.htm"
OUTPUT_FILE = "data/tbk_dataset.json"

class TurkishStatuteRAG:
    def __init__(self):
        self.chunks = []
        # Metadata storage for current parsing state
        self.current_part = "Başlangıç"
        self.current_chapter = "Genel Hükümler"
        
        # Metadata for the Law itself (parsed from header)
        self.law_meta = {
            "acceptance_date": "",
            "journal_date": "",
            "journal_no": ""
        }

    # --- PHASE A: FETCHING (The Raw Source) ---
    def fetch_statute(self):
        print(f"Fetching {LAW_URL}...")
        try:
            # Fake headers to look like a browser
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(LAW_URL, headers=headers)
            # Mevzuat.gov.tr uses windows-1254 (Turkish Windows) encoding
            response.encoding = 'windows-1254'
            return response.text
        except Exception as e:
            print(f"Error fetching URL: {e}")
            return None

    # --- PHASE B: STRUCTURING (PIC & Hierarchy) [cite: 1, 3] ---
    def parse_html(self, html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        
        # 1. Parse Header Table for Dates (Kabul Tarihi, etc.)
        self._parse_law_header(soup)

        # 2. Iterate through paragraphs to find Parts, Chapters, and Articles
        # Mevzuat usually uses <p> tags with specific classes or inline styles
        paragraphs = soup.find_all('p')
        
        current_article_no = None
        current_article_title = ""
        fikra_counter = 0

        # Regex patterns
        part_pattern = re.compile(r"^(BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ|ALTINCI|YEDİNCİ)\s+KISIM", re.IGNORECASE)
        chapter_pattern = re.compile(r"^(BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ)\s+BÖLÜM", re.IGNORECASE)
        # Matches "MADDE 12 -" or "MADDE 12 :" or "MADDE 12"
        article_pattern = re.compile(r"^MADDE\s+(\d+)", re.IGNORECASE)

        for p in paragraphs:
            text = p.get_text(strip=True).replace('\xa0', ' ')
            if not text:
                continue

            # Update Hierarchy Context
            if part_pattern.match(text):
                self.current_part = text
                continue
            if chapter_pattern.match(text):
                self.current_chapter = text
                continue

            # Check for Article Start
            art_match = article_pattern.match(text)
            if art_match:
                current_article_no = art_match.group(1)
                fikra_counter = 1 # Reset fıkra count
                
                # Cleanup text: remove "MADDE 12 -" prefix to get the actual rule
                # Sometimes the first paragraph is on the same line as header
                clean_text = re.sub(r"^MADDE\s+\d+[\s\-\.:]*", "", text).strip()
                
                if clean_text:
                    self._create_chunk(current_article_no, fikra_counter, clean_text)
                    fikra_counter += 1
            
            # Check for Continuation (Subsequent Paragraphs/Fıkra)
            elif current_article_no and not part_pattern.match(text) and not chapter_pattern.match(text):
                # Heuristic: If it looks like a normal sentence, it's the next Fıkra
                # Ignoring page numbers or random metadata often found in footer
                if len(text) > 5 and not text.startswith("Sayfa"): 
                    self._create_chunk(current_article_no, fikra_counter, text)
                    fikra_counter += 1

    def _parse_law_header(self, soup):
        # Mevzuat.gov.tr usually puts dates at the top of the page
        try:
            # Get all text from the page for searching
            page_text = soup.get_text()
            
            # Regex patterns adjusted for mevzuat.gov.tr format:
            # "Kabul Tarihi : 11/1/2011"
            # "Tarih : 4/2/2011" or "Tarih  : 4/2/2011"  
            # "Sayı : 27836" or "Sayı  : 27836"
            kabul_match = re.search(r"Kabul\s+Tarihi\s*:\s*([\d/\.]+)", page_text)
            rg_tarih_match = re.search(r"Tarih\s*:\s*([\d/\.]+)", page_text)
            rg_sayi_match = re.search(r"Sayı\s*:\s*(\d+)", page_text)

            if kabul_match: self.law_meta['acceptance_date'] = kabul_match.group(1)
            if rg_tarih_match: self.law_meta['journal_date'] = rg_tarih_match.group(1)
            if rg_sayi_match: self.law_meta['journal_no'] = rg_sayi_match.group(1)
            
            print(f"Parsed dates: {self.law_meta}")
        except Exception as e:
            print(f"Warning: Could not parse header dates. Error: {e}")

    # --- PHASE C & D: ENRICHMENT (SAC, Poly-Vector, Graph) [cite: 2, 4, 5] ---
    def _create_chunk(self, article_no, fikra_no, text):
        
        # 1. Generate Stable ID (Hierarchy Feature)
        chunk_id = f"LAW:{LAW_NO}-{article_no}-{fikra_no}"

        # 2. SAC: Generate Parent Summary (Simulated LLM Call)
        parent_summary = self._mock_llm_summarize(article_no, text)

        # 3. Poly-Vector: Generate Citation Variations
        citations = [
            f"TBK {LAW_NO} Madde {article_no}",
            f"TBK m.{article_no}",
            f"{LAW_NO} S.K. m.{article_no}",
            f"Türk Borçlar Kanunu {article_no}. Madde"
        ]

        # 4. GraphRAG: Extract implicit links (Regex)
        links = self._extract_graph_links(text)

        # Construct the final JSON Object
        chunk_obj = {
            "id": chunk_id,
            "text": text,
            "metadata": {
                "doc_type": "statute",
                "hierarchy": {
                    "law_no": LAW_NO,
                    "law_title": "Türk Borçlar Kanunu",
                    "part": self.current_part,
                    "chapter": self.current_chapter,
                    "article": str(article_no),
                    "paragraph": str(fikra_no)
                },
                "dates": {
                    "acceptance_date": self.law_meta.get('acceptance_date', ''),
                    "journal_date": self.law_meta.get('journal_date', ''),
                    "journal_no": self.law_meta.get('journal_no', '')
                },
                "urls": {
                    "source_url": LAW_URL
                },
                "sac_context": {
                    "parent_summary": parent_summary
                },
                "poly_vector": {
                    "citation_label": f"TBK {LAW_NO} Madde {article_no} Fıkra {fikra_no}",
                    "citation_variations": citations
                },
                "graph_links": links
            }
        }
        
        self.chunks.append(chunk_obj)

    def _mock_llm_summarize(self, article_no, text):
        # IN PRODUCTION: Replace this with openai.ChatCompletion.create(...)
        # Prompt: "Summarize the legal intent of this article in one sentence."
        return f"Bu madde (Md. {article_no}), ilgili borçlar hukuku kuralını ve uygulama şartlarını düzenler."

    def _extract_graph_links(self, text):
        # Finds references to other laws and articles in Turkish legal texts
        links = []
        
        # Pattern 1: "X sayılı Kanun" (e.g., "4721 sayılı Kanun")
        for match in re.finditer(r"(\d+)\s+sayılı\s+(?:Kanun|Kanunu)", text, re.IGNORECASE):
            ref_law_no = match.group(1)
            links.append({
                "target_id": f"LAW:{ref_law_no}",
                "relation_type": "statutory_reference",
                "text_span": match.group(0)
            })
        
        # Pattern 2: "X sayılı Kanun Hükmünde Kararname" (KHK)
        for match in re.finditer(r"(\d+)\s+sayılı\s+Kanun\s+Hükmünde\s+Kararname", text, re.IGNORECASE):
            ref_khk_no = match.group(1)
            links.append({
                "target_id": f"KHK:{ref_khk_no}",
                "relation_type": "khk_reference",
                "text_span": match.group(0)
            })
        
        # Pattern 3: Article references like "X inci/üncü/nci madde" 
        for match in re.finditer(r"(\d+)\s*(?:inci|üncü|nci|ncı|uncu|ıncı)\s+madde", text, re.IGNORECASE):
            ref_article = match.group(1)
            links.append({
                "target_id": f"LAW:{LAW_NO}-{ref_article}",
                "relation_type": "internal_reference",
                "text_span": match.group(0)
            })
        
        # Pattern 4: "bu Kanunun X inci maddesi" - internal reference
        for match in re.finditer(r"bu\s+Kanunun\s+(\d+)", text, re.IGNORECASE):
            ref_article = match.group(1)
            links.append({
                "target_id": f"LAW:{LAW_NO}-{ref_article}",
                "relation_type": "internal_reference",
                "text_span": match.group(0)
            })
        
        # Pattern 5: "TMK m. X" or "TBK m. X" style references
        for match in re.finditer(r"(TMK|TBK|TCK|HMK|İİK)\s*m\.?\s*(\d+)", text, re.IGNORECASE):
            law_abbrev = match.group(1).upper()
            ref_article = match.group(2)
            law_no_map = {"TMK": "4721", "TBK": "6098", "TCK": "5237", "HMK": "6100", "İİK": "2004"}
            target_law_no = law_no_map.get(law_abbrev, "UNKNOWN")
            links.append({
                "target_id": f"LAW:{target_law_no}-{ref_article}",
                "relation_type": "cross_reference",
                "text_span": match.group(0)
            })
            
        return links

    def save_json(self):
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(self.chunks)} chunks to {OUTPUT_FILE}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    processor = TurkishStatuteRAG()
    
    # Phase A
    html = processor.fetch_statute()
    
    if html:
        # Phase B, C, D
        processor.parse_html(html)
        processor.save_json()
        
        # Print the specific example from the user request (Article 12)
        for chunk in processor.chunks:
            if chunk["metadata"]["hierarchy"]["article"] == "12":
                print(json.dumps(chunk, ensure_ascii=False, indent=2))
                break