# AI Assignment #3 — Process Log

**Course:** CISC3024 Pattern Recognition  
**Student:** CHE CHI HIN, Angus (UC325182)  
**Rule:** no hand-written code; AI agent (Cursor) performs search / implementation / experiments / drafting.  
**Human role:** set goals, approve direction (RSSPCA + new repo), request fact-check fixes, submit Moodle PDF.

---

## How I directed the AI agent

1. Asked Cursor to take full ownership of Ass3 and clarify constraints.
2. Chose algorithm **A: RSSPCA (2025, PLOS One)** over 2DPCA+LDA and Robust Sparse LDA.
3. Chose delivery like Ass1/Ass2: **new independent repo** + English LaTeX PDF + `process_log.md` + public GitHub.
4. After first PDF draft, asked for online fact-check; then ordered **all** fixes: paper protocol, RSMPCA, η₂=1, z-score, 3-seed recognition, polluted-fit reconstruction.

---

## Chronological technical log

- Searched 2022–2025 PCA/LDA CV work; locked Wang et al. RSSPCA (PLOS One 2025).
- Created `CISC3024AI3`; sklearn Olivetti CDN 403 → Cambridge AT&T ORL.
- First run: mean-center only, clean-train/dirty-test, single seed → weak robustness story.
- Fact-check vs paper: polluted train, z-score, 56×46, Table-1 η, RSMPCA, 3 recognition seeds.
- Re-ran: salt-pepper polluted-fit RSMPCA/RSSPCA ≈23.65 dB vs PCA 22.96; recognition r=5 RSSPCA 0.900±0.012.
- Report rebuilt as Ass2-style `report.tex` → tectonic → `AIAssignment3_Report.pdf`.

## Key commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/ensure_orl.py
python run_all.py
# PDF (needs tectonic binary):
./_tectonic -X compile report.tex && cp report.pdf AIAssignment3_Report.pdf
```

## Submission checklist

- [x] Recent PCA/LDA algorithm (RSSPCA, PLOS One 2025)
- [x] CV/PR application (face reconstruction + recognition)
- [x] Report sections (1)–(6)
- [x] Paper-aligned protocol notes + multi-seed
- [x] Source code webpage link
- [x] No student-written code (Cursor agent only)
