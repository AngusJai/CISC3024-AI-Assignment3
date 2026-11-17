# CISC3024 AI Assignment #3

**Robust Sparse Smooth PCA (RSSPCA) for Face Reconstruction and Recognition**  
CHE CHI HIN, Angus · UC325182

**Report (submit):** [`AIAssignment3_Report.pdf`](./AIAssignment3_Report.pdf)  
**Source:** https://github.com/AngusJai/CISC3024-AI-Assignment3  

Paper: Wang et al., *PLOS One*, 2025. DOI: https://doi.org/10.1371/journal.pone.0323281  
Official MATLAB ref: https://github.com/yuzhounh/RSSPCA

## Layout

```
├── AIAssignment3_Report.pdf
├── report.tex / process_log.md
├── rsspca/           # PCA, PCA-L1, RSPCA, RSMPCA, RSSPCA
├── data/             # ORL loader (56×46, z-score)
├── experiments/      # polluted-fit reconstruction + multi-seed recognition
├── scripts/ensure_orl.py
├── run_all.py
└── outputs/          # figures + metrics JSON
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/ensure_orl.py
python run_all.py
```

Defaults follow Wang et al. Table 1: `η1=0.001`, `η2=1` (RSSPCA), RSMPCA `η2=10^(-0.2)`.  
Recognition averages seeds `0 1 2`. Reconstruction fits on 320 clean + 80 polluted images.

## Headline results (approx.)

| Setting | PCA | RSSPCA / RSMPCA |
|--------|----:|----------------:|
| Salt-pepper polluted-fit PSNR @ r=30 | ~23.0 dB | **~23.6 dB** |
| 1-NN @ r=5 (3-seed mean) | ~0.89 | **~0.90 (RSSPCA)** |
| 1-NN @ r=40 (3-seed mean) | ~0.98 | ~0.98 |

## Rule

No student-written code — Cursor AI agent performed algorithm search, implementation, experiments, and report drafting under student direction.
