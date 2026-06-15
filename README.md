# Intuit Explainable ML Hackathon (NYC 2026)

This repository contains the end-to-end machine learning pipeline developed for the Intuit Techweek NYC Explainable ML Hackathon. The goal of this project is to predict small and medium-sized business (SMB) loan behaviors while maintaining strict model explainability and providing actionable counterfactuals.

## 🗂 Repository Structure

*   `src/`: Contains all core pipeline scripts (`build_spine.py`, `build_C.py`, `generate_D.py`, etc.).
*   `data/`: Local data directory (raw datasets are gitignored; see `data_dictionary.csv` for schema).
*   `expected_ids/`: Manifests and query IDs required for pipeline validation.
*   `submission/`: Final generated CSVs for decisions, trajectories, and counterfactuals.
*   `docs/`: Hackathon brief and schedule for project context.

## ⚙️ Methodology

Our approach bridges the gap between predictive power and financial transparency, broken down into the following core phases:

### 1. Feature Engineering & Spine Generation (`build_spine.py`)
*   **Data Aggregation:** Processed raw financial transactions, standardizing temporal anomalies and handling missing data points.
*   **Feature Construction:** Extracted key financial health indicators (e.g., observed monthly revenue, overdraft counts, cash balance trends) to create a robust modeling spine ready for training.

### 2. Decision & Trajectory Modeling (Submissions A & B)
*   **Decision Engine:** Developed an underwriting model to predict approval likelihood and initial risk scoring, utilizing a high-performance classification architecture.
*   **Cohort Forecasting:** Modeled the grid of probabilities over time to forecast long-term SMB financial trajectories, predicting repayment behaviors and default risks across subsequent weeks.

### 3. Counterfactual Generation (`build_C.py`)
*   **Actionable Adjustments:** Implemented an algorithmic approach to generate mathematically sound counterfactuals. For any declined application, the pipeline computes the precise minimum adjustments required—such as modifying the requested loan amount or improving average monthly revenue—to flip the decision boundary to an approval.


## 🚀 Setup & Execution

**1. Install Dependencies**
\`\`\`bash
pip install -r requirements.txt
\`\`\`

**2. Run the Pipeline**
Execute the pipeline sequentially from the `src/` directory:
\`\`\`bash
python src/step0_check.py
python src/build_spine.py
python src/build_submissions.py
\`\`\`

**3. Validate Output**
\`\`\`bash
python src/validate_submission.py
\`\`\`

## 🔮 Next Steps
To make these insights actionable for end-users, the next logical step involves wrapping this inference pipeline in a REST API and deploying a full-stack dashboard. This would allow loan officers to dynamically interact with the counterfactuals and adjust parameters in real-time.
