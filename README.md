# CISC3024 AI Assignment #3

**Robust Sparse Smooth PCA (RSSPCA) for Face Reconstruction and Recognition**  
CHE CHI HIN, Angus · UC325182

**Report (submit):** [`AIAssignment3_Report.pdf`](./AIAssignment3_Report.pdf)  
**Source:** https://github.com/AngusJai/CISC3024-AI-Assignment3  

Paper: Wang et al., *PLOS One*, 2025. DOI: https://doi.org/10.1371/journal.pone.0323281  
Official MATLAB ref: https://github.com/yuzhounh/RSSPCA

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/ensure_orl.py
python scripts/ensure_yale.py
PYTHONPATH=. python run_all.py          # ORL + extras (ablation, Yale)
# or skip extras: PYTHONPATH=. python run_all.py --skip-extras
```

## What is in the report

1. How AI found RSSPCA among recent PCA/LDA options  
2. Algorithm (MM update, Laplacian smoothness, RSPCA/RSMPCA special cases)  
3. Implementation (Python NumPy/SciPy; no student-written code)  
4. Experiments: polluted-fit recon, 3-seed recognition, η ablation, Yale second set  
5. Lessons learnt  
6. Source link  

## Headline results

| Setting | PCA | Smooth methods |
|--------|----:|---------------:|
| ORL salt-pepper PSNR @ r=30 | ~23.0 | **~23.6 (RSMPCA/RSSPCA)** |
| ORL 1-NN @ r=5 (3-seed) | ~0.89 | **~0.90 (RSSPCA)** |
| Yale salt-pepper PSNR @ r=20 | ~20.4 | **~21.3 (RSMPCA)** |

## Rule

No student-written code — Cursor AI agent did search, coding, experiments, and report drafting under student direction.
