## Analysis on the User Study - Adaptive VR for Different Anchors for Different Conditions for Different Tasks

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run analyses

- Questionnaire (requires `Questionnaire.csv` in the project root):

```bash
python scripts/run_analysis.py --csv "Questionnaire.csv" --outdir outputs/questionnaire
```

- Logs (requires `StudyLogs/` with participant folders):

```bash
python scripts/run_logs_analysis.py --logs_root StudyLogs --outdir outputs/logs
```

### Reports

- Questionnaire report: `outputs/questionnaire/report.md`
- Logs report: `outputs/logs/report.md`

### Outputs

- Tables (CSVs) and figures (PNGs) are written under:
  - `outputs/questionnaire/` (figures in `outputs/questionnaire/figs/`)
  - `outputs/logs/` (figures in `outputs/logs/figs/`)

### Project structure

- `scripts/`: entry points (`run_analysis.py`, `run_logs_analysis.py`)
- `cix_analysis/`: questionnaire analysis modules
- `logs_analysis/`: logs analysis modules
- `outputs/`: generated results (reports, CSVs, figures)
- `Questionnaire.csv`: questionnaire data (input)
- `StudyLogs/`: raw log data (input)
