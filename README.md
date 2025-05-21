# 🧬 Predictive Analytics for Hepatitis C: Enhancing Patient Care Using Data Science Techniques

This project leverages machine learning to classify individuals as either **blood donors** or **Hepatitis C patients** based on laboratory medical test results. The objective is to assist in early and accurate detection of Hepatitis C, fibrosis, and cirrhosis, thereby improving patient care outcomes.

## 📂 Project Structure

- `hepatisis_c_predictive_analytics.ipynb` – Main Colab notebook containing the full pipeline from data preprocessing to model evaluation.
- `README.md` – Project overview and usage instructions.

## 📊 Dataset

- Source: [UCI Machine Learning Repository – HCV Data](https://archive.ics.uci.edu/dataset/571/hcv+data)
- Features: Laboratory results (ALT, AST, ALP, CHOL, PROT, etc.)
- Target: Patient class – Blood Donor or Hepatitis C (including fibrosis and cirrhosis)
- Size: 615 instances

## ⚙️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Seaborn, Matplotlib
- KNN Imputer
- Logistic Regression, Random Forest, SVM, KNN

## 🛠️ ML Pipeline

1. **Data Preprocessing**
   - Missing value imputation using **KNN Imputer**
   - Label encoding of target classes
   - Class consolidation: Hepatitis (1, 2, 3) vs. Blood Donor (0)
   - Train-test split with stratification

2. **Model Training**
   - Trained and evaluated multiple classifiers:
     - Logistic Regression
     - Random Forest
     - Support Vector Machine (SVM)
     - K-Nearest Neighbors (KNN)

3. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - ROC Curve analysis
   - Classification reports
   - Model comparison

## ✅ Key Results

- **Logistic Regression (Binary Classification)**:
  - Accuracy: **97%**
  - Precision: 1.00 (Hepatitis), 0.97 (Donors)
  - F1-Score: 0.89 (Hepatitis), 0.98 (Donors)

## 💡 Highlights

- Efficient handling of **class imbalance**
- High performance in classifying early-stage liver diseases
- Easy-to-understand ML pipeline with clinical relevance
- Model-ready for integration in decision support systems

## 🚀 Future Work

- Apply explainable AI (SHAP/LIME) for interpretability
- Explore deep learning models (e.g., MLP)
- Build a Streamlit dashboard for real-time prediction
