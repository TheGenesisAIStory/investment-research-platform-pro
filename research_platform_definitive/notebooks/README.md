# Canonical Notebook Index

This folder is intentionally an index, not a second notebook source tree.

Use these canonical notebooks:

| Domain | Canonical notebook |
|---|---|
| Company valuation | `../company_valuation/notebooks/Company_Valuation_Final_Version.ipynb` |
| Portfolio research | `../portfolio_analysis/notebooks/Portfolio-Analysis-Model_RESEARCH_PLATFORM_PRO.ipynb` |
| ML equity / stock lab | `../machine_learning_lab/notebooks/ML_Stock_Lab_Experiments.ipynb` |
| Data/API operations | `../data_api_management/notebooks/Data_API_Management_Colab.ipynb` |

Why no duplicate notebooks here:

- notebook execution paths are used by Streamlit orchestration and validators;
- duplicate copies drift quickly and create confusing Colab errors;
- stale or generated copies are kept in `../archive/` only for traceability.

Reference notebooks from the old project live in `../laboratorio/` and are not source of truth until promoted into one of the canonical notebooks or a reusable `src/` module.
