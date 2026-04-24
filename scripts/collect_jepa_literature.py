#!/usr/bin/env python3
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

BASE_DIR = Path('/root/workspace/jepa')
PDF_DIR = BASE_DIR / 'pdfs'
META_DIR = BASE_DIR / 'metadata'
HF_DIR = BASE_DIR / 'hf_pages'
SCRIPT_DIR = BASE_DIR / 'scripts'

ARXIV_API = 'http://export.arxiv.org/api/query?search_query={query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending'
ATOM = '{http://www.w3.org/2005/Atom}'
ARXIV = '{http://arxiv.org/schemas/atom}'
UA = 'Roger-JEPA-Literature-Collector/1.0 (contact: local-openclaw-agent)'

QUERIES = [
    'all:"JEPA"',
    'all:"Joint Embedding Predictive Architecture"',
    'all:"Joint-Embedding Predictive Architecture"',
    'all:"I-JEPA"',
    'all:"V-JEPA"',
    'all:"V-JEPA2"',
    'all:"M3-JEPA"',
    'all:"MC-JEPA"',
    'all:"LeJEPA"',
    'all:"LLM-JEPA"',
    'all:"C-JEPA"',
    'all:"D-JEPA"',
    'all:"JEPA-Reasoner"',
]

KEYWORD_PATTERNS = [
    re.compile(r'\bjepa\b', re.I),
    re.compile(r'joint[- ]embedding predictive architecture', re.I),
    re.compile(r'\bi-jepa\b', re.I),
    re.compile(r'\bv-jepa\b', re.I),
    re.compile(r'\bv-jepa2\b', re.I),
    re.compile(r'\bm3-jepa\b', re.I),
    re.compile(r'\bmc-jepa\b', re.I),
    re.compile(r'\blejepa\b', re.I),
    re.compile(r'\bllm-jepa\b', re.I),
    re.compile(r'\bc-jepa\b', re.I),
    re.compile(r'\bd-jepa\b', re.I),
    re.compile(r'\bjepa-reasoner\b', re.I),
]


def fetch(url: str, timeout: int = 30) -> bytes:
    req = Request(url, headers={'User-Agent': UA})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def text_or_empty(node, path: str) -> str:
    child = node.find(path)
    return child.text.strip() if child is not None and child.text else ''


def parse_entry(entry) -> Dict:
    raw_id = text_or_empty(entry, f'{ATOM}id').split('/abs/')[-1]
    paper_id = re.sub(r'v\d+$', '', raw_id)
    title = re.sub(r'\s+', ' ', text_or_empty(entry, f'{ATOM}title'))
    summary = re.sub(r'\s+', ' ', text_or_empty(entry, f'{ATOM}summary'))
    authors = [re.sub(r'\s+', ' ', text_or_empty(author, f'{ATOM}name')) for author in entry.findall(f'{ATOM}author')]
    categories = [cat.attrib.get('term', '') for cat in entry.findall(f'{ATOM}category')]
    pdf_url = ''
    for link in entry.findall(f'{ATOM}link'):
        if link.attrib.get('title') == 'pdf':
            pdf_url = link.attrib.get('href', '')
            break
    if pdf_url and not pdf_url.endswith('.pdf'):
        pdf_url = pdf_url + '.pdf'
    if not pdf_url:
        pdf_url = f'https://arxiv.org/pdf/{paper_id}.pdf'
    return {
        'id': paper_id,
        'title': title,
        'summary': summary,
        'authors': authors,
        'published': text_or_empty(entry, f'{ATOM}published'),
        'updated': text_or_empty(entry, f'{ATOM}updated'),
        'categories': categories,
        'pdf_url': pdf_url,
        'abs_url': f'https://arxiv.org/abs/{paper_id}',
    }


def relevant(paper: Dict) -> bool:
    blob = f"{paper['title']}\n{paper['summary']}"
    return any(p.search(blob) for p in KEYWORD_PATTERNS)


def query_arxiv(query: str, max_results: int = 100) -> List[Dict]:
    url = ARXIV_API.format(query=quote(query), max_results=max_results)
    data = fetch(url, timeout=45)
    root = ET.fromstring(data)
    out = []
    for entry in root.findall(f'{ATOM}entry'):
        paper = parse_entry(entry)
        if relevant(paper):
            out.append(paper)
    return out


def check_hf_page(paper_id: str) -> Tuple[bool, str]:
    url = f'https://huggingface.co/papers/{paper_id}'
    try:
        body = fetch(url, timeout=20)
        text = body[:4000].decode('utf-8', errors='ignore')
        lower = text.lower()
        if 'page not found' in lower or '404' in lower[:500]:
            return False, url
        return True, url
    except HTTPError as e:
        if e.code == 404:
            return False, url
        return False, url
    except URLError:
        return False, url
    except Exception:
        return False, url


def download_pdf(paper: Dict) -> str:
    target = PDF_DIR / f"{paper['id']}.pdf"
    if target.exists() and target.stat().st_size > 0:
        return 'existing'
    try:
        data = fetch(paper['pdf_url'], timeout=120)
        target.write_bytes(data)
        return 'downloaded'
    except Exception as e:
        return f'error: {e}'


def main() -> int:
    for d in [BASE_DIR, PDF_DIR, META_DIR, HF_DIR, SCRIPT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    all_papers: Dict[str, Dict] = {}
    query_hits: Dict[str, List[str]] = {}

    for query in QUERIES:
        try:
            papers = query_arxiv(query)
        except Exception as e:
            print(f'[warn] query failed: {query}: {e}', file=sys.stderr)
            continue
        query_hits[query] = [p['id'] for p in papers]
        for paper in papers:
            existing = all_papers.get(paper['id'])
            if existing:
                existing['matched_queries'] = sorted(set(existing['matched_queries'] + [query]))
            else:
                paper['matched_queries'] = [query]
                all_papers[paper['id']] = paper
        time.sleep(1.0)

    papers = sorted(all_papers.values(), key=lambda p: p['published'], reverse=True)

    for idx, paper in enumerate(papers, start=1):
        hf_ok, hf_url = check_hf_page(paper['id'])
        paper['hf_papers_available'] = hf_ok
        paper['hf_papers_url'] = hf_url
        paper['pdf_status'] = download_pdf(paper)
        paper['local_pdf'] = str(PDF_DIR / f"{paper['id']}.pdf")
        print(f"[{idx}/{len(papers)}] {paper['id']} | hf={hf_ok} | pdf={paper['pdf_status']} | {paper['title']}")
        time.sleep(0.5)

    json_path = META_DIR / 'papers.json'
    csv_path = META_DIR / 'papers.csv'
    queries_path = META_DIR / 'queries.json'
    md_path = BASE_DIR / 'README.md'

    json_path.write_text(json.dumps(papers, indent=2, ensure_ascii=False), encoding='utf-8')
    queries_path.write_text(json.dumps(query_hits, indent=2, ensure_ascii=False), encoding='utf-8')

    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'id', 'published', 'title', 'authors', 'categories', 'hf_papers_available',
            'hf_papers_url', 'abs_url', 'pdf_url', 'local_pdf', 'matched_queries'
        ])
        for paper in papers:
            writer.writerow([
                paper['id'], paper['published'], paper['title'], '; '.join(paper['authors']),
                '; '.join(paper['categories']), paper['hf_papers_available'],
                paper['hf_papers_url'], paper['abs_url'], paper['pdf_url'],
                paper['local_pdf'], '; '.join(paper['matched_queries'])
            ])

    lines = [
        '# JEPA literature collection',
        '',
        f'- Total papers collected: **{len(papers)}**',
        f'- PDFs directory: `{PDF_DIR}`',
        f'- Metadata JSON: `{json_path}`',
        f'- Metadata CSV: `{csv_path}`',
        '',
        '## Query set',
        '',
    ]
    lines.extend([f'- `{q}` → {len(query_hits.get(q, []))} hits' for q in QUERIES])
    lines.extend(['', '## Papers', ''])
    for paper in papers:
        authors = ', '.join(paper['authors'][:4]) + (' et al.' if len(paper['authors']) > 4 else '')
        hf = 'yes' if paper['hf_papers_available'] else 'no'
        lines.append(f"- **{paper['published'][:10]}** | `{paper['id']}` | {paper['title']} | {authors} | HF papers: {hf}")
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(f'\nDone. Collected {len(papers)} papers into {BASE_DIR}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
