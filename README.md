# Bank Customer Churn & Risk Analysis

## Overview

This project focuses on predicting and analyzing bank customer churn using Machine Learning, Exploratory Data Analysis (EDA), statistical analysis, SHAP explainability, and an interactive Dash dashboard.

The project started with the Bank Customer Churn dataset obtained from Kaggle. The dataset was further expanded and prepared as part of the project, resulting in a final dataset of 165,034 customer records used for the analysis.

The main objective is to identify customers who are at higher risk of leaving the bank and provide an interpretable risk-scoring system that can support customer retention strategies.

## Dataset

The initial dataset was obtained from Kaggle:

Kaggle – Bank Customer Churn Prediction  
https://www.kaggle.com/code/kmalit/bank-customer-churn-prediction

The Kaggle dataset was used as the starting point and was further expanded/prepared for this project.

Final dataset used for analysis:

- Total records: 165,034
- Retained customers: 130,113 (78.8%)
- Churned customers: 34,921 (21.2%)
- Features: 14

The dataset contains customer information such as:

- Credit Score
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card ownership
- Active Membership
- Estimated Salary
- Geography
- Churn/Exited status

## Project Workflow

Kaggle Dataset  
→ Dataset Expansion and Preparation  
→ Data Cleaning  
→ Exploratory Data Analysis  
→ Statistical Analysis  
→ Machine Learning Models  
→ Model Tuning  
→ Class Imbalance Handling  
→ Threshold Optimization  
→ SHAP Explainability  
→ Customer Risk Scoring  
→ Interactive Dashboard

## Exploratory Data Analysis

The project performs EDA to understand customer churn patterns.

Analysis includes:

- Overall churn distribution
- Churn by geography
- Churn by gender
- Churn by active membership
- Churn by age
- Churn by number of products
- Numerical feature distributions
- Correlation analysis
- Heatmaps
- Risk profile analysis
- Geographic analysis

Overall churn rate was 21.2%.

Some important patterns observed were:

- Germany showed a much higher churn rate than France and Spain.
- Inactive members had considerably higher churn than active members.
- Customers with certain age groups and product counts showed higher churn risk.
- Age and number of products were among the strongest factors associated with churn.

## Statistical Analysis

Statistical techniques were used to investigate the relationship between customer characteristics and churn.

The project includes:

- Chi-square tests
- Mann-Whitney U tests
- Cramér's V
- Rank-biserial effect size
- Confidence intervals

Because the dataset is large, statistical significance alone was not treated as evidence of practical importance. Effect sizes were also considered.

## Machine Learning Models

Three classification models were evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

### Baseline Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.833 | 0.694 | 0.381 | 0.492 | 0.814 |
| Random Forest | 0.859 | 0.724 | 0.542 | 0.620 | 0.875 |
| XGBoost | 0.864 | 0.735 | 0.560 | 0.636 | 0.887 |

XGBoost achieved the best baseline performance across the main evaluation metrics and was selected for further improvement.

## Handling Class Imbalance

Two approaches were compared:

- SMOTE
- Class weighting

Class weighting performed better for the project's main objective of detecting churners.

The tuned class-weighted model achieved approximately 0.792 recall during validation compared with approximately 0.561 using the tuned SMOTE approach.

Therefore, class weighting was selected for the final model.

## Model Tuning

The XGBoost model was further optimized using:

- Randomized hyperparameter search
- 30 trials
- 5-fold cross-validation
- PR-AUC as the tuning metric

The dataset was split using an 80/20 stratified split with seed 42.

- Training records: 132,027
- Test records: 33,007

The final test set was kept separate and was used only for the final evaluation.

## Threshold Optimization

Instead of using the default classification threshold of 0.50, the project selected a threshold of 0.63.

The threshold was selected using out-of-fold training predictions to obtain the highest precision while maintaining recall of at least 0.70.

This increased the model's ability to identify potential churners.

## Final Model Results

| Metric | Baseline XGBoost | Final Model |
|---|---:|---:|
| Accuracy | 0.864 | 0.851 |
| Precision | 0.735 | 0.633 |
| Recall | 0.560 | 0.701 |
| F1 Score | 0.636 | 0.665 |
| ROC-AUC | 0.887 | 0.890 |

The most important improvement was churn recall:

**0.560 → 0.701**

This represents an improvement of approximately **14.1 percentage points**.

The reduction in accuracy and precision was an intentional trade-off to identify more customers who were likely to churn.

## Confusion Matrix

The final model was evaluated on 33,007 previously unseen test customers.

- True Negatives: 23,184
- False Positives: 2,839
- False Negatives: 2,089
- True Positives: 4,895

The final recall was 0.701 and precision was 0.633.

Compared with the baseline, the final model detected more churners while producing more false alarms.

## Customer Risk Scoring

The model assigns customers into different risk levels based on their predicted churn score.

The selected threshold of 0.63 is used for churn-risk flagging.

Risk concentration was also analyzed:

- Riskiest 5%: 91.8% precision and captures 21.7% of churners
- Riskiest 10%: 84.1% precision and captures 39.8% of churners
- Riskiest 20%: 68.1% precision and captures 64.4% of churners

This demonstrates how the model can help prioritize customers for retention efforts.

## SHAP Explainability

SHAP was used to explain the model's predictions and identify the features that most influenced churn-risk scores.

The most influential features included:

- Number of Products
- Age
- Active Membership
- Gender
- Geography
- Balance

SHAP explanations describe what the model relies on for its predictions and should not be interpreted as proof of causal relationships.

## Risk Profiles

The project also identifies combinations of customer characteristics associated with different churn-risk levels.

Examples include:

- Higher-risk profiles involving Germany, older age groups, inactive membership, and certain product counts.
- Lower-risk profiles involving France, younger/middle age groups, active membership, and selected product counts.

The project uses these profiles to help interpret model predictions and support potential retention analysis.

## Interactive Dashboard

An interactive dashboard was developed using Plotly Dash.

The dashboard supports:

- Individual customer risk prediction
- CSV upload
- Input validation
- Churn-risk score
- Risk tier
- Churn flag
- SHAP-based prediction explanations
- Customer filtering
- Risk-profile analysis
- Heatmaps
- Ranked customer profiles

The trained model and preprocessing pipeline are saved using Joblib so that predictions can be reproduced through the dashboard.

## Dashboard Validation

Several validation tests were performed.

Example:

- A valid customer profile was successfully scored.
- Invalid countries were rejected.
- Invalid ages below 18 were rejected.
- Large-scale scoring was tested with 1,000,000 customers in approximately 1.4 seconds.

The saved model reproduced the reported test performance:

- Recall: 0.701
- Precision: 0.633
- ROC-AUC: 0.890

## Research Paper Comparison

The project uses the research paper:

**“Predicting Bank Customer Churn: An XGBoost Approach to Enhancing Customer Retention”**

by Ruchika Bhuria and Srinivas Aluvala as a benchmark/reference.

The project's baseline XGBoost model achieved a churn recall of approximately 0.560, consistent with the benchmark result used in the project.

The final tuned and threshold-optimized model increased churn recall to 0.701 while maintaining a similar ROC-AUC.

Therefore, the project extends the baseline analysis by focusing on improved churn detection, class imbalance handling, threshold optimization, and explainable customer risk scoring.

## Project Structure

Bank-Customer-Churn-Risk-Analysis/

- Bank_Churn_Dashboard.ipynb
- Bank_Churn_Project_Review_Updated.pptx
- requirements.txt
- README.md

## Technologies Used

- Python
- Pandas
- NumPy
- SciPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Imbalanced-learn
- SHAP
- Plotly
- Dash
- Joblib
- Jupyter Notebook

## How to Run

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook Bank_Churn_Dashboard.ipynb
```

Run the notebook cells to perform the analysis and model development.

The Dash dashboard can then be launched from the dashboard code included in the notebook.

## Presentation

The project presentation is included in the repository:

`Bank_Churn_Project_Review_Updated.pptx`

It contains the project methodology, EDA findings, model comparison, tuning process, final results, SHAP analysis, risk profiles, dashboard, and limitations.

## Limitations

- The analysis is based on a single bank customer dataset.
- The original dataset was obtained from Kaggle and subsequently expanded/prepared for this project.
- The data origin beyond the Kaggle starting point is not independently verified.
- Customer IDs are not fully unique in the final dataset.
- The model's risk score is intended for ranking and risk classification and should not automatically be interpreted as a true probability of churn.
- Recall is lower for some younger age groups.
- Suggested retention actions have not been tested on real customers.
- The model has not been externally validated on datasets from other banks.

## Future Improvements

Future work could include:

- Testing the model on datasets from other banks
- Adding transaction and customer-service data
- Incorporating historical customer behavior
- Testing real retention offers
- Monitoring model performance after deployment
- Building a production-ready API
- Adding automated model retraining

## Conclusion

This project demonstrates an end-to-end approach to bank customer churn risk analysis.

Starting from a Kaggle bank customer churn dataset, the dataset was expanded and prepared for analysis, followed by EDA, statistical analysis, machine learning, class imbalance handling, hyperparameter tuning, threshold optimization, SHAP explainability, and interactive dashboard development.

The final model improved churn recall from 0.560 to 0.701, allowing the system to identify a larger proportion of potential churners while providing interpretable risk information for customer retention analysis.

**Author:** D R Sanjanaa
