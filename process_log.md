# AI Assignment #3 — Process Log

**Course:** CISC3024 Pattern Recognition  
**Student:** CHE CHI HIN, Angus (UC325182)  
**Rule:** no hand-written code; AI agent (Cursor) performs search / implementation / experiments / drafting.  
**Human role:** set goals, approve algorithm/repo choices, request fact-check + polish rounds, submit Moodle PDF.

---

## How I directed the AI agent (prompt history, condensed)

### Round 1 — kickoff
> 上次做完 ai ass1 ass2 而家做3 全權交俾你負責… AI Assignment #3… recent PCA/LDA… NOT allowed to write any code by yourself.

Agent planned options; I chose **A: RSSPCA (2025)** and **new repo like Ass1**.

### Round 2 — first PDF unreadable
> 優化下pdf 咩都睇唔到  
> references hw2的格式再做

Agent rebuilt `report.tex` with Ass2 LaTeX/fancyhdr/booktabs and compiled via tectonic.

### Round 3 — fact-check
> 上網fact check下有無啲咩問題 同埋睇下成份hw 有咩優缺點要改  
> 全都改動

Fixes: RSMPCA, η₂=1, z-score, 56×46, polluted-fit, 3-seed recognition, report honesty about protocol.

### Round 4 — polish 1–5
> 或者你曬佢1-5

Agent added: η ablation, eigenface compare strip, Yale second dataset, relative-error curves, longer process log.

---

## Chronological technical log

1. Literature search → locked Wang et al., PLOS One 2025 (RSSPCA).
2. New repo `CISC3024AI3`; sklearn Olivetti CDN 403 → Cambridge AT&T ORL.
3. First implementation (mean-center, clean-train) → weak robustness story.
4. Fact-check vs paper → polluted-fit, z-score, Table-1 η, RSMPCA, 3 seeds.
5. Ass2-style LaTeX PDF via tectonic.
6. Extras: `eta_ablation.png`, `eigenfaces_compare.png`, `reconstruction_rel_error_curves.png`, classic Yale (GitHub mirror; official CDNs 403).

## Headline numbers (final)

| Setting | PCA | Best smooth/sparse |
|--------|----:|-------------------:|
| ORL salt-pepper polluted-fit PSNR @ r=30 | 22.96 | **RSMPCA 23.65** |
| ORL 1-NN @ r=5 (3-seed) | 0.886±0.010 | **RSSPCA 0.900±0.012** |
| Yale salt-pepper polluted-fit PSNR @ r=20 | 20.40 | **RSMPCA 21.35** |

## Key commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/ensure_orl.py
python scripts/ensure_yale.py
PYTHONPATH=. python run_all.py
# PDF:
./_tectonic -X compile report.tex && cp report.pdf AIAssignment3_Report.pdf
```

## Submission checklist

- [x] Recent PCA/LDA (RSSPCA, PLOS One 2025)
- [x] CV/PR application (face recon + recognition)
- [x] Report sections (1)–(6) + Ass2 LaTeX format
- [x] Paper-aligned protocol + multi-seed + Yale + η ablation
- [x] Source code webpage link
- [x] No student-written code
