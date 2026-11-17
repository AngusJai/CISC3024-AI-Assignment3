# AI Assignment #3 — Process Log

**Course:** CISC3024 Pattern Recognition  
**Student:** CHE CHI HIN, Angus (UC325182)  
**Rule:** no hand-written code; AI agent (Cursor) performs search / implementation / experiments / drafting.  
**Human role:** set goals, approve direction (RSSPCA + new repo), verify outputs, submit Moodle PDF.

---

## How I directed the AI agent

1. Asked Cursor to take full ownership of Ass3 and clarify constraints.
2. Chose algorithm option **A: RSSPCA (2025, PLOS One)** over 2DPCA+LDA and Robust Sparse LDA.
3. Chose delivery like Ass1: **new independent repo** + English PDF + `process_log.md` + public GitHub link.
4. Agent searched recent PCA/LDA CV papers, locked Wang et al. RSSPCA, reimplemented in Python, ran ORL face experiments, wrote the report.

---

## Chronological technical log

- Searched 2022–2025 PCA/LDA CV work; shortlisted FF-PCA-LDA, 2DPCA+LDA (2024), Robust Sparse LDA (PR 2024), **RSSPCA (2025)**.
- Selected RSSPCA: recent, PCA-family, clear robust+sparse+smooth story, public paper + MATLAB reference (`yuzhounh/RSSPCA`).
- Created sibling project `CISC3024AI3` (not mixed into Ass1 FasterNet tree).
- Implemented NumPy/SciPy solvers: PCA, PCA-L1, RSPCA, RSSPCA (MM update + spatial Laplacian).
- sklearn Olivetti download blocked (HTTP 403); switched to Cambridge **AT&T ORL** (`att_faces.zip`), resized to 64×64.
- Fixed multi-component stability with Gram–Schmidt re-orthogonalization before deflation.
- Tuned `(η1, η2) = (0.001, 0.1)` via small recognition sweep.
- Ran reconstruction (clean / block occlusion / salt-pepper) and 1-NN recognition vs #components.
- Drafted `AIAssignment3_Report.md` → PDF; pushed GitHub repo for §6 link.

## Key commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/ensure_orl.py   # if ORL not cached
python run_all.py
```

## Submission checklist

- [x] Recent PCA/LDA algorithm (RSSPCA, PLOS One 2025)
- [x] CV/PR application (face reconstruction + recognition)
- [x] Report sections (1)–(6)
- [x] Source code webpage link
- [x] No student-written code (Cursor agent only)
