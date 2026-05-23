# Investment Research Platform Pro

The cleaned canonical project lives in:

```text
research_platform_definitive/
```

Start here:

```bash
cd research_platform_definitive
python3 scripts/validate_definitive_bundle.py
streamlit run research_platform_app/app.py
```

Common Data/API control app from the repository root:

```bash
streamlit run research_platform_definitive/data_api_app.py
```

Refresh Data/API contracts after first setup:

```bash
python3 research_platform_definitive/scripts/migrate_data_api_control.py
```

Canonical notebooks:

- `research_platform_definitive/company_valuation/notebooks/Company_Valuation_Final_Version.ipynb`
- `research_platform_definitive/portfolio_analysis/notebooks/Portfolio-Analysis-Model_RESEARCH_PLATFORM_PRO.ipynb`
- `research_platform_definitive/machine_learning_lab/notebooks/ML_Stock_Lab_Experiments.ipynb`
- `research_platform_definitive/data_api_management/notebooks/Data_API_Management_Colab.ipynb`

Current project status:

- `research_platform_definitive/docs/PROJECT_STATUS_2026-05-23.md`
- `research_platform_definitive/docs/PROJECT_STATUS_FINAL.md`

Populate the final local research database and prompt packet:

```bash
python3 research_platform_definitive/scripts/populate_research_database.py
python3 research_platform_definitive/scripts/llm_lab_cli.py packet
```

Key final docs:

- `research_platform_definitive/docs/DATA_CENTER_OPERATING_MODEL.md`
- `research_platform_definitive/docs/FORMULE_STRATEGIE_REASONING.md`
- `research_platform_definitive/docs/LLM_LAB_VIBE_TRADING.md`
- `research_platform_definitive/docs/MOBILE_QUANTDINGER_HANDOFF.md`
- `research_platform_definitive/docs/ACADEMIC_METHODS.md`
- `research_platform_definitive/docs/MEMORY_RECOVERY_PLAN.md`

Local databases not kept on Google Drive were preserved under:

```text
research_platform_definitive/local_databases_not_on_drive/
```
