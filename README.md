# Disease Prediction System 🫀

## Project Overview
- Course: Data Warehousing and Data Mining (CSE_4060)
- Activity-Based Learning (ABL) demonstration project
- Predicts presence of heart disease using the UCI Heart Disease dataset
- Tools used: Python, scikit-learn, pandas, matplotlib, seaborn

## Team Members
- Naman Alex Xavier — namanalex@gmail.com
- Joel Josy — joeljosy449@gmail.com

## Dataset
- Name: Heart Disease UCI Dataset
- Source: UCI Machine Learning Repository / Kaggle
- Rows: 303 | Features: 13 | Target: Binary (0 = No Disease, 1 = Disease)
- Features include: age, sex, chest pain type, resting blood pressure, cholesterol, max heart rate, etc.

## Project Structure
- disease_prediction.py — main script
- heart.csv — dataset
- requirements.txt — dependencies
- All .png output files:
  - target_value_count.png
  - age_distribution.png
  - correlation_heatmap.png
  - confusion_matrix.png
  - feature_importance.png
  - model_comparison.png
  - roc_curve.png
  - age_vs_target_boxplot.png

## How to Run
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure heart.csv is in the same folder
4. Run the main script:
   ```bash
   python disease_prediction.py
   ```
5. All output charts will be saved as PNG files in the same directory

## Models Used & Results
| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression | ~XX% | Linear baseline model with strong interpretability |
| Random Forest (n_estimators=100) | ~XX% | Ensemble model, also used for feature importance |
| K-Nearest Neighbors (k=5) | ~XX% | Distance-based classifier using standardized features |

Note: Replace ~XX% with exact values after running the script.

## Key Findings
- Top 3 most important features identified by Random Forest
- Best performing model identified by comparison
- Dataset is balanced (165 disease / 138 no disease)

## Future Work
- Try deep learning models (Neural Networks)
- Deploy as a web app using Flask or Streamlit
- Expand dataset for better generalization
- Add more diseases beyond heart disease

## License
MIT License
