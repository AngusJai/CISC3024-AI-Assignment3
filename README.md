# CISC3024 AI Assignment #3

**Robust Sparse Smooth PCA (RSSPCA) for Face Reconstruction and Recognition**  
CHE CHI HIN, Angus · UC325182

**Report (submit):** [`AIAssignment3_Report.pdf`](./AIAssignment3_Report.pdf)  
**Source:** https://github.com/AngusJai/CISC3024-AI-Assignment3  

Paper: Wang et al., *Robust sparse smooth principal component analysis for face reconstruction and recognition*, PLOS One, 2025.  
DOI: https://doi.org/10.1371/journal.pone.0323281 · Official MATLAB ref: https://github.com/yuzhounh/RSSPCA

## Layout

```
├── AIAssignment3_Report.pdf
├── process_log.md
├── rsspca/           # PCA, PCA-L1, RSPCA, RSSPCA (NumPy/SciPy)
├── data/             # ORL / Olivetti-style loader
├── experiments/      # reconstruction + recognition
├── scripts/ensure_orl.py
├── run_all.py
└── outputs/          # figures + metrics JSON
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/ensure_orl.py    # downloads AT&T ORL faces if needed
python run_all.py
```

Dataset: AT&T ORL (400 faces, 40 subjects), resized to 64×64.  
Split: 7 train / 3 test per subject. Defaults: `η1=0.001`, `η2=0.1`.

## Expected results (approx.)

| Setting | PCA | RSSPCA |
|--------|----:|-------:|
| Clean recon PSNR @ r=40 | ~23.3 dB | ~23.2 dB |
| 1-NN accuracy @ r=10 | ~0.95 | ~0.97 |
| 1-NN accuracy @ r=40 | ~0.975 | ~0.967 |

## Rule

No student-written code — Cursor AI agent performed algorithm search, implementation, experiments, and report drafting under student direction.
