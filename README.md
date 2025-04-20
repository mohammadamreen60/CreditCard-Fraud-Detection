 Credit Card Fraud Detection

This project is a machine learning-based solution for detecting fraudulent transactions using the famous Kaggle Credit Card Fraud Detection dataset.

## 📁 Project Structure

PROJECT/ │ ├── creditcard.py # Main script for data processing, modeling, and evaluation 
           ├── cdd.csv # Dataset file └── README.md # Project documentation
Dataset

The dataset used is the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), which contains transactions made by European cardholders in September 2013. It presents transactions that occurred in two days, with 492 frauds out of 284,807 transactions.

- Features: V1 to V28 (PCA transformed)
- Other columns: `Time`, `Amount`, `Class` (1 = Fraud, 0 = Not Fraud)

## 🚀 What This Project Does

- Loads and analyzes the dataset
- Handles class imbalance using techniques like undersampling/oversampling (if used)
- Trains a classification model (e.g., Logistic Regression, Random Forest, etc.)
- Evaluates with metrics like Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Displays confusion matrix or other visualizations
## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn (if used)
- Jupyter Notebook / VS Code

## ✅ Results
### 🔹 Support Vector Machine (SVM)
- **Accuracy**: 93%
- **Precision**: 0.95 (Fraud)
- **Recall**: 0.90 (Fraud)
- **F1-Score**: 0.93 (Fraud)

  ### 🔹 K-Nearest Neighbors (KNN)
- **Accuracy**: 93%
- **Precision**: 1.00 (Non-Fraud), 0.88 (Fraud)
- **Recall**: 0.84 (Non-Fraud), 1.00 (Fraud)
- **F1-Score**: 0.91 (Non-Fraud), 0.93 (Fraud)
- **Confusion Matrix**:

  ## 🙋‍♀️ Author

**Amreen Fathima**  
[GitHub Profile](https://github.com/mohammadamreen60)
