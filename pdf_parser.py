"""PDF → structured text pipeline with caching."""

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pdfplumber
import pypdf


class PDFParser:
    """Parse a folder of PDFs into structured chapters."""

    CHAPTER_PATTERNS = [
        re.compile(r"^(?:Chapter|CHAPTER)\s+(\d+|[IVX]+)[:.\s]*(.*)$"),
        re.compile(r"^\s*(\d+)[:.\s]+([A-Z][A-Za-z\s]+)\s*$"),
        re.compile(r"^\s*Part\s+(\d+|[IVX]+)[:.\s]*(.*)$", re.IGNORECASE),
    ]
    SPEAKER_PATTERNS = [
        re.compile(r"^([A-Z][a-zA-Z\s\-']{1,20}):\s*(.*)$"),
        re.compile(r"([A-Z][a-zA-Z\s\-']{1,20})\s+(said|replied|asked|shouted|whispered|cried|yelled)\s*:\s*([^\.]+)"),
    ]

    def __init__(self, cache_dir: str = "./book_parsed"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _file_hash(path: str) -> str:
        h = hashlib.sha256()
        h.update(Path(path).read_bytes())
        return h.hexdigest()[:16]

    def _load_cache(self, key: str) -> Optional[Dict]:
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))
        return None

    def _save_cache(self, key: str, data: Dict):
        cache_file = self.cache_dir / f"{key}.json"
        cache_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def extract_text_pdfplumber(self, path: str) -> str:
        parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
        return "\n\n".join(parts)

    def extract_text_pypdf(self, path: str) -> str:
        parts = []
        with open(path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
        return "\n\n".join(parts)

    def extract_text(self, path: str) -> str:
        try:
            return self.extract_text_pdfplumber(path)
        except Exception as exc:
            print(f"[PDFParser] pdfplumber failed ({exc}), falling back to pypdf")
            return self.extract_text_pypdf(path)

    def split_chapters(self, text: str) -> Dict[str, Dict]:
        lines = text.splitlines()
        chapters: Dict[str, List[str]] = {}
        current_title = "Preface"
        current_lines: List[str] = []

        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            matched = False
            for pat in self.CHAPTER_PATTERNS:
                m = pat.match(line)
                if m:
                    if current_lines:
                        chapters.setdefault(current_title, []).extend(current_lines)
                    current_title = line
                    current_lines = []
                    matched = True
                    break
            if not matched:
                current_lines.append(line)

        if current_lines:
            chapters.setdefault(current_title, []).extend(current_lines)

        # If no chapters detected, split into two rough halves
        if len(chapters) <= 1:
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            if len(paragraphs) > 1:
                mid = len(paragraphs) // 2
                chapters = {
                    "Part 1": paragraphs[:mid],
                    "Part 2": paragraphs[mid:],
                }
            else:
                chapters = {"Full Text": paragraphs}

        return {
            title: {"title": title, "text": "\n".join(lines)}
            for title, lines in chapters.items()
        }

    def detect_speakers(self, text: str) -> Dict[str, List[str]]:
        speakers: Dict[str, List[str]] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            for pat in self.SPEAKER_PATTERNS:
                for m in pat.finditer(line):
                    groups = m.groups()
                    name = groups[0]
                    dialogue = groups[-1]
                    speakers.setdefault(name, []).append(dialogue.strip())
        return speakers

    def parse_folder(self, folder: str) -> Dict[str, Dict]:
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        pdfs = sorted(folder.glob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError(f"No PDFs found in {folder}")

        all_chapters: Dict[str, Dict] = {}
        for pdf in pdfs:
            key = f"{pdf.stem}_{self._file_hash(str(pdf))}"
            cached = self._load_cache(key)
            if cached is not None:
                print(f"[PDFParser] Cache hit: {pdf.name}")
                for ch_name, ch_data in cached.items():
                    all_chapters[f"{pdf.stem} / {ch_name}"] = ch_data
                continue

            print(f"[PDFParser] Parsing: {pdf.name}")
            text = self.extract_text(str(pdf))
            chapters = self.split_chapters(text)
            for ch_data in chapters.values():
                ch_data["speakers"] = self.detect_speakers(ch_data["text"])
                ch_data["source_pdf"] = pdf.name

            self._save_cache(key, chapters)
            for ch_name, ch_data in chapters.items():
                all_chapters[f"{pdf.stem} / {ch_name}"] = ch_data

        return all_chapters


def main():
    parser = PDFParser()
    result = parser.parse_folder("./book")
    print(f"Parsed {len(result)} chapters")
    for k in list(result.keys())[:5]:
        print(f"  - {k}: {len(result[k]['text'])} chars")


if __name__ == "__main__":
    main()
