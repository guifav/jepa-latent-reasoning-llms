#!/usr/bin/env python3
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path('/root/workspace/jepa')
META = BASE / 'metadata'
ANALYSIS = BASE / 'analysis'
ANALYSIS.mkdir(parents=True, exist_ok=True)

papers = json.loads((META / 'papers.json').read_text(encoding='utf-8'))
core_ids = {p['id'] for p in json.loads((META / 'papers_core.json').read_text(encoding='utf-8'))}

PRIMARY_THEMES = [
    ('foundations_theory_objective', [
        r'\blejepa\b', r'\bvar-jepa\b', r'\bsparsejepa\b', r'\bkerjepa\b', r'\blpjepa\b',
        r'provable', r'heuristics', r'gaussian', r'variational', r'kernel', r'quasimetric',
        r'normalization', r'feature normalization', r'auxiliary', r'contrastive',
        r'from alignment to prediction', r'compute-efficient', r'density', r'weak-sigreg', r'jepamatch'
    ]),
    ('benchmarks_surveys_critiques', [
        r'benchmark', r'empirical study', r'controlled study', r'analysis', r'what drives',
        r'philosophical', r'compare', r'comparative evaluation', r'revisiting', r'pretext matters', r'convergence analysis'
    ]),
    ('language_reasoning', [
        r'\bllm\b', r'reason', r'talker', r'token', r'\bbert-jepa\b', r'language-invariant',
        r'language-conditioned', r'neural tokenizer', r'text fusion', r'text-image'
    ]),
    ('audio_speech', [r'audio', r'speech', r'waveform', r'listen', r'foley']),
    ('multimodal_vlm_vla', [r'multimodal', r'vision-language', r'cross-modal', r'visual-language', r'm3-jepa', r'mumo', r'\bvla\b']),
    ('medical_bio_health', [r'medical', r'clinical', r'radiology', r'ultrasound', r'eeg', r'fmri', r'surgical', r'health', r'ehr', r'dna', r'genomic', r'single-cell', r'parkinson']),
    ('scientific_industrial_domains', [r'remote sensing', r'geospatial', r'earth observation', r'satellite', r'wireless', r'6g', r'csi', r'collider', r'sonar', r'mine', r'multiphysics', r'physical systems', r'engine emissions', r'molecular graphs', r'jet ']),
    ('world_models_planning_control', [r'world model', r'planning', r'controller', r'control', r'navigation', r'robot', r'policy', r'autonomous driving', r'action-conditioned', r'manipulation', r'latent dynamics', r'\brl\b', r'reinforcement learning', r'dreamer']),
    ('video_temporal', [r'\bv-jepa', r'video', r'temporal', r'motion', r'facial expression', r'action dataset', r'dashcam', r'tube']),
]

SECONDARY_TAGS = {
    'theory': [r'provable', r'theory', r'theoretical', r'formal', r'gaussian', r'variational', r'kernel', r'energy-based'],
    'video': [r'video', r'v-jepa', r'temporal', r'motion'],
    'audio': [r'audio', r'speech', r'waveform', r'listen'],
    'language': [r'\bllm\b', r'language', r'text', r'token', r'reason'],
    'multimodal': [r'multimodal', r'vision-language', r'cross-modal', r'visual-language', r'vla'],
    'world_model': [r'world model', r'planning', r'control', r'navigation', r'policy', r'robot', r'rl', r'reinforcement learning'],
    'medical': [r'medical', r'clinical', r'radiology', r'ultrasound', r'eeg', r'fmri', r'surgical', r'health', r'ehr', r'dna', r'genomic', r'single-cell'],
    'remote_or_science': [r'remote sensing', r'geospatial', r'earth observation', r'wireless', r'6g', r'csi', r'physics', r'multiphysics', r'sonar', r'molecular', r'collider'],
    'generative_bridge': [r'generative', r'diffusion', r'generator', r'generation', r'tokenizer', r'denoising'],
    'benchmarking': [r'benchmark', r'empirical study', r'controlled study', r'analysis', r'what drives', r'compare', r'revisiting', r'pretext matters'],
}

THEME_DESCRIPTIONS = {
    'vision_perception_ssl': 'Core visual/perceptual JEPA work and generic representation learning papers that do not fit a more specific vertical.',
    'video_temporal': 'Video, motion, temporal prediction and spatiotemporal JEPA work.',
    'audio_speech': 'Speech and audio representation learning with JEPA objectives.',
    'multimodal_vlm_vla': 'Multimodal, vision-language and vision-language-action JEPA extensions.',
    'language_reasoning': 'Language, latent reasoning and token-generation decoupling work.',
    'world_models_planning_control': 'World models, planning, control, robotics, RL and embodied decision-making.',
    'foundations_theory_objective': 'Objective design, theory, regularization and fundamental training recipes for JEPA.',
    'medical_bio_health': 'Clinical, biomedical, neuro, genomics and healthcare JEPA applications.',
    'scientific_industrial_domains': 'Remote sensing, wireless, physics, industrial and other scientific-domain JEPA applications.',
    'benchmarks_surveys_critiques': 'Benchmarking, comparisons, ablations, evaluations and critique papers.'
}

MILESTONES = [
    ('2301.08243', 'Foundational image JEPA baseline; establishes the starting point for the whole literature.'),
    ('2307.12698', 'First major temporal extension: JEPA for motion/content disentanglement in video.'),
    ('2311.15830', 'Early modality expansion from vision into audio.'),
    ('2408.07514', 'Backbone diversification beyond ViT-heavy formulations into CNNs.'),
    ('2409.05929', 'Multimodal JEPA enters the literature in a dedicated form.'),
    ('2410.03755', 'Generative bridge appears via denoising-oriented JEPA framing.'),
    ('2410.19560', 'The literature starts connecting JEPA to adjacent SSL theory such as contrastive learning.'),
    ('2412.10925', 'JEPA becomes a broader video representation-learning family, not just a single paper line.'),
    ('2501.14622', 'JEPA enters policy representation learning and RL-oriented work.'),
    ('2505.03176', 'Autoregressive / sequential world-model direction becomes explicit.'),
    ('2506.09985', 'V-JEPA 2 marks a step from representation learning toward understanding, prediction and planning.'),
    ('2507.02915', 'Audio JEPA becomes a more explicit standalone program of work.'),
    ('2509.14252', 'JEPA ideas move decisively into LLM territory.'),
    ('2510.00974', 'Image-generation bridge appears through JEPA + text fusion.'),
    ('2511.08544', 'LeJEPA consolidates the theory/recipe turn with a cleaner and more principled training story.'),
    ('2512.10942', 'Vision-language JEPA becomes explicit rather than implicit or ad hoc.'),
    ('2512.19171', 'Latent reasoning is framed as a first-class JEPA research direction.'),
    ('2602.10098', 'JEPA begins to plug directly into VLA systems.'),
    ('2602.11389', 'Object-centric causal/world-model JEPA becomes explicit and technically grounded.'),
    ('2603.19312', 'End-to-end JEPA world models from pixels push toward unified latent world modeling.'),
    ('2603.22281', 'JEPA starts being paired with stronger reasoning stacks for latent planning.'),
    ('2604.21046', 'Recent work returns to representation shaping, indicating the objective-design frontier remains open.'),
]

GAPS = [
    {
        'gap_id': 'G1',
        'gap_name': 'No common evaluation spine across JEPA modalities',
        'priority': 'high',
        'evidence': 'Only 15 title-level benchmark/evaluation/critique papers in a 176-paper corpus, while applications sprawl across vision, video, audio, medical, remote sensing, robotics and wireless.',
        'why_it_matters': 'Claims are hard to compare across papers because datasets, metrics, and baselines vary wildly by vertical.',
        'opportunity': 'Build a JEPA benchmark pack with standardized baselines, compute budgets, and modality-specific plus cross-modal evaluation slices.',
        'representative_papers': '2404.08471, 2512.24497, 2602.13507, 2603.22649, 2604.10514'
    },
    {
        'gap_id': 'G2',
        'gap_name': 'Language and latent reasoning are promising but still immature',
        'priority': 'high',
        'evidence': 'Language/reasoning papers appear mostly from late 2025 onward; only 12 papers land in this bucket with title-based taxonomy.',
        'why_it_matters': 'This is the most interesting bridge from JEPA into agentic systems and modern LLM use-cases, but the area is still recipe-fragile.',
        'opportunity': 'Prototype JEPA-based latent reasoning with stronger evaluation on reasoning, planning, and token-generation robustness.',
        'representative_papers': '2509.14252, 2512.19171, 2603.22281, 2601.00366'
    },
    {
        'gap_id': 'G3',
        'gap_name': 'Generative JEPA remains fragmented',
        'priority': 'high',
        'evidence': 'The corpus contains multiple bridge attempts—denoising, diffusion, text fusion, tokenizer-style, variational and reasoner-style papers—but no single dominant recipe.',
        'why_it_matters': 'JEPA is strong at representation learning, but the lack of a stable generative interface limits adoption in mainstream generative AI workflows.',
        'opportunity': 'Study a unified JEPA-to-generator stack that preserves predictive latent semantics while enabling controllable generation.',
        'representative_papers': '2410.03755, 2509.14252, 2510.00974, 2512.19171, 2603.20111'
    },
    {
        'gap_id': 'G4',
        'gap_name': 'Training recipe stability is still unsettled',
        'priority': 'high',
        'evidence': 'Objective/theory papers accelerate only in late 2025–2026, with repeated focus on Gaussian geometry, sparsity, auxiliary losses, SIGReg-like regularization and normalization.',
        'why_it_matters': 'The field is still paying recipe tax; too much work goes into making JEPA stable instead of building on a settled core recipe.',
        'opportunity': 'Reproduce and distill a small set of stable JEPA recipes across image, video and language settings.',
        'representative_papers': '2511.08544, 2602.01456, 2603.05924, 2603.20111, 2604.21046'
    },
    {
        'gap_id': 'G5',
        'gap_name': 'World-model papers are growing fast, but transfer/generalization is under-proven',
        'priority': 'high',
        'evidence': 'World-model/planning/control is one of the fastest-growing clusters, but most papers target narrow environments such as driving, navigation, robotics, wireless control or physics simulators.',
        'why_it_matters': 'Without cross-domain transfer evidence, JEPA world models may remain a collection of local wins instead of a general latent planning paradigm.',
        'opportunity': 'Test whether one JEPA world-model backbone can transfer across video prediction, manipulation, driving and navigation with minimal task-specific surgery.',
        'representative_papers': '2505.03176, 2506.09985, 2602.11389, 2602.10098, 2603.19312'
    },
    {
        'gap_id': 'G6',
        'gap_name': 'Audio and non-vision modalities are still thinly explored',
        'priority': 'medium',
        'evidence': 'Only 5 audio/speech papers are caught by title-based taxonomy, far below vision/video growth.',
        'why_it_matters': 'If JEPA is genuinely a general predictive-learning principle, the audio/speech story should be deeper and better benchmarked by now.',
        'opportunity': 'Run a focused scaling and benchmark study for audio/speech JEPA, ideally tied to the same objective-design questions explored in vision.',
        'representative_papers': '2311.15830, 2507.02915, 2509.23238, 2512.07168'
    },
    {
        'gap_id': 'G7',
        'gap_name': 'Scientific and industrial applications are numerous but siloed',
        'priority': 'medium',
        'evidence': 'The corpus shows scattered JEPA use in remote sensing, wireless, physics, sonar, genomics, and molecular graphs, but little evidence of shared abstraction layers or transfer recipes.',
        'why_it_matters': 'A common JEPA recipe for structured scientific data could unlock much broader reuse.',
        'opportunity': 'Abstract the common pattern behind domain papers into a reusable JEPA template for non-natural-signal data.',
        'representative_papers': '2412.05333, 2502.03933, 2602.17162, 2603.25216, 2604.01349'
    }
]


def classify_primary(title: str) -> str:
    blob = title.lower()
    for theme, patterns in PRIMARY_THEMES:
        for pat in patterns:
            if re.search(pat, blob):
                return theme
    return 'vision_perception_ssl'


def secondary_tags(title: str, summary: str):
    blob = f'{title} {summary}'.lower()
    out = []
    for tag, patterns in SECONDARY_TAGS.items():
        if any(re.search(p, blob) for p in patterns):
            out.append(tag)
    return out


def phase_for_date(date: str) -> str:
    if date < '2024-01-01':
        return 'phase_1_foundations'
    if date < '2025-01-01':
        return 'phase_2_diversification'
    if date < '2025-07-01':
        return 'phase_3_embodied_expansion'
    if date < '2026-01-01':
        return 'phase_4_language_theory_multimodal'
    return 'phase_5_world_models_reasoning_applications'


for p in papers:
    p['corpus_group'] = 'core' if p['id'] in core_ids else 'adjacent'
    p['primary_theme'] = classify_primary(p['title'])
    p['secondary_tags'] = secondary_tags(p['title'], p['summary'])
    p['year'] = p['published'][:4]
    p['month'] = p['published'][:7]
    p['phase'] = phase_for_date(p['published'][:10])

# taxonomy csv
with (ANALYSIS / 'taxonomy_by_theme.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['paper_id', 'published', 'year', 'month', 'corpus_group', 'primary_theme', 'secondary_tags', 'title', 'authors', 'hf_papers_available', 'arxiv_url', 'local_pdf'])
    for p in sorted(papers, key=lambda x: (x['published'], x['id'])):
        w.writerow([
            p['id'], p['published'], p['year'], p['month'], p['corpus_group'], p['primary_theme'], ';'.join(p['secondary_tags']),
            p['title'], '; '.join(p['authors']), p['hf_papers_available'], p['abs_url'], p['local_pdf']
        ])

# taxonomy summary
summary_rows = []
for theme in THEME_DESCRIPTIONS:
    theme_rows = [p for p in papers if p['primary_theme'] == theme]
    theme_core = sum(1 for p in theme_rows if p['corpus_group'] == 'core')
    theme_adj = sum(1 for p in theme_rows if p['corpus_group'] == 'adjacent')
    years = Counter(p['year'] for p in theme_rows)
    first = min(theme_rows, key=lambda x: x['published']) if theme_rows else None
    summary_rows.append({
        'theme': theme,
        'description': THEME_DESCRIPTIONS[theme],
        'count_total': len(theme_rows),
        'count_core': theme_core,
        'count_adjacent': theme_adj,
        'first_appearance': first['published'][:10] if first else '',
        'first_paper_id': first['id'] if first else '',
        'first_paper_title': first['title'] if first else '',
        'count_2023': years.get('2023', 0),
        'count_2024': years.get('2024', 0),
        'count_2025': years.get('2025', 0),
        'count_2026': years.get('2026', 0),
    })

with (ANALYSIS / 'taxonomy_summary.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    w.writeheader()
    w.writerows(summary_rows)

# timeline counts
month_counts = Counter(p['month'] for p in papers)
months = sorted(month_counts)
cumulative = 0
with (ANALYSIS / 'timeline_counts.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['month', 'papers_published', 'cumulative_papers'])
    for m in months:
        cumulative += month_counts[m]
        w.writerow([m, month_counts[m], cumulative])

# timeline milestones
id_to_paper = {p['id']: p for p in papers}
with (ANALYSIS / 'timeline_milestones.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['published', 'paper_id', 'title', 'phase', 'significance'])
    for pid, sig in MILESTONES:
        p = id_to_paper[pid]
        w.writerow([p['published'][:10], pid, p['title'], p['phase'], sig])

# gap map csv
with (ANALYSIS / 'gap_map.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['gap_id', 'gap_name', 'priority', 'evidence', 'why_it_matters', 'opportunity', 'representative_papers'])
    w.writeheader()
    w.writerows(GAPS)

# markdown reports
lines = []
lines.append('# JEPA timeline')
lines.append('')
lines.append(f'- Corpus size: **{len(papers)}** papers')
lines.append(f'- Core JEPA set: **{len(core_ids)}** papers')
lines.append(f'- Adjacent set: **{len(papers) - len(core_ids)}** papers')
lines.append('')
lines.append('## Quantitative growth')
lines.append('')
by_year = Counter(p['year'] for p in papers)
for year in sorted(by_year):
    lines.append(f'- {year}: **{by_year[year]}** papers')
lines.append('')
lines.append('## Phase view')
lines.append('')
phase_labels = {
    'phase_1_foundations': '2023 — foundations and first modality extensions',
    'phase_2_diversification': '2024 — diversification into backbones, multimodality and conceptual bridges',
    'phase_3_embodied_expansion': '2025 H1 — world models, policy learning and broader applications',
    'phase_4_language_theory_multimodal': '2025 H2 — language, theory, generation and multimodal expansion',
    'phase_5_world_models_reasoning_applications': '2026 YTD — world-model consolidation, latent reasoning and application explosion',
}
for phase, label in phase_labels.items():
    count = sum(1 for p in papers if p['phase'] == phase)
    lines.append(f'- {label}: **{count}** papers')
lines.append('')
lines.append('## Milestones')
lines.append('')
for pid, sig in MILESTONES:
    p = id_to_paper[pid]
    lines.append(f"- **{p['published'][:10]}** — `{pid}` — **{p['title']}**. {sig}")
(ANALYSIS / 'timeline.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

lines = []
lines.append('# JEPA taxonomy by theme')
lines.append('')
lines.append('Method note: primary themes are assigned by transparent title-first keyword rules; secondary tags are added from title+abstract metadata. This makes the taxonomy auditable, but not equivalent to full-text human coding.')
lines.append('')
for row in sorted(summary_rows, key=lambda r: (-r['count_total'], r['theme'])):
    lines.append(f"## {row['theme']}")
    lines.append('')
    lines.append(f"- Description: {row['description']}")
    lines.append(f"- Total papers: **{row['count_total']}** (core: {row['count_core']}, adjacent: {row['count_adjacent']})")
    lines.append(f"- First appearance: {row['first_appearance']} — `{row['first_paper_id']}` — {row['first_paper_title']}")
    lines.append(f"- Year split: 2023={row['count_2023']}, 2024={row['count_2024']}, 2025={row['count_2025']}, 2026={row['count_2026']}")
    examples = [p for p in papers if p['primary_theme'] == row['theme']][:5]
    lines.append('- Representative examples:')
    for p in examples:
        lines.append(f"  - `{p['id']}` — {p['title']}")
    lines.append('')
(ANALYSIS / 'taxonomy.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

lines = []
lines.append('# JEPA gap map')
lines.append('')
lines.append('These gaps are derived from the corpus distribution, milestone ordering, and concentration patterns across the title/abstract metadata. They are meant to identify promising work fronts, not to claim the field is missing only these items.')
lines.append('')
for gap in GAPS:
    lines.append(f"## {gap['gap_id']} — {gap['gap_name']}")
    lines.append('')
    lines.append(f"- Priority: **{gap['priority']}**")
    lines.append(f"- Evidence from corpus: {gap['evidence']}")
    lines.append(f"- Why it matters: {gap['why_it_matters']}")
    lines.append(f"- Opportunity: {gap['opportunity']}")
    lines.append(f"- Representative papers: {gap['representative_papers']}")
    lines.append('')
(ANALYSIS / 'gap_map.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

lines = []
lines.append('# JEPA landscape overview')
lines.append('')
lines.append(f'- Total corpus: **{len(papers)}** papers')
lines.append(f'- Core JEPA papers: **{len(core_ids)}**')
lines.append(f'- Adjacent papers: **{len(papers) - len(core_ids)}**')
lines.append(f'- Earliest paper in corpus: **{min(papers, key=lambda x: x["published"])["published"][:10]}**')
lines.append(f'- Most recent paper in corpus: **{max(papers, key=lambda x: x["published"])["published"][:10]}**')
lines.append('')
lines.append('## Main takeaways')
lines.append('')
lines.append('- The field starts in image-centric self-supervision, then expands quickly into motion/video, audio and multimodality.')
lines.append('- The steepest recent growth comes from world models, planning, JEPA-to-generation bridges, and theory/recipe papers.')
lines.append('- Language/reasoning is recent and still clearly early-stage compared with vision/video.')
lines.append('- Application growth is outpacing benchmarking discipline, which creates a comparison problem across papers.')
lines.append('')
lines.append('## Files')
lines.append('')
for name in ['timeline.md', 'timeline_counts.csv', 'timeline_milestones.csv', 'taxonomy.md', 'taxonomy_by_theme.csv', 'taxonomy_summary.csv', 'gap_map.md', 'gap_map.csv']:
    lines.append(f'- `{name}`')
(ANALYSIS / 'overview.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

print('analysis files written to', ANALYSIS)
