# Contributing to TRUST-BC

Thanks for your interest in this project. TRUST-BC is primarily a research artifact
accompanying a manuscript, but contributions that improve reproducibility, extend the
benchmark, or fix bugs are welcome.

## Ways to contribute

- **Bug reports / reproducibility issues:** please open a GitHub issue with your OS,
  Python version, and the exact traceback or discrepancy you observed.
- **External validation:** if you run TRUST-BC on a new cohort (e.g., an external
  European registry such as METABRIC), we would be glad to hear about it — see the
  "External Geographic Validation" item in the manuscript's Limitations section.
- **Methodological extensions:** e.g., additional calibration diagnostics, alternative
  conformal scores, or deep tabular baselines are welcome as pull requests.

## Development setup

```bash
git clone https://github.com/RezaAbolghasemi/TRUST-BC-Multicohort-Benchmark.git
cd TRUST-BC-Multicohort-Benchmark
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # use requirements-revision.txt for the scripts in scripts/
```

Before opening a pull request:

1. Make sure `python -m py_compile src/trust_bc_multicohort_benchmark.py scripts/*.py` passes.
2. If you change the pipeline logic, re-run it end-to-end on at least the WBCD cohort
   (which requires no manual data download) and confirm the output tables are sane.
3. Keep changes to `results/` (the curated, paper-matching outputs) out of unrelated
   pull requests — new runs should be written to `results/run_<timestamp>/`, which is
   git-ignored by default.

## Code style

The pipeline is a single, linearly-organized script (`src/trust_bc_multicohort_benchmark.py`) mirroring the
structure described in the manuscript's Methods section. Please keep new code
consistent with the existing style (type hints where practical, docstrings on public
functions) rather than introducing a different framework or package layout.

## Questions

For questions about the methodology itself, please refer first to the manuscript
(`paper/TRUST-BC_manuscript_v1.1.docx`) and Online Resource 1 (`supplementary/`), which document every design decision in detail.
