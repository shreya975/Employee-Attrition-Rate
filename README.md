# Employee Attrition Intelligence 📊

An interactive **Employee Attrition Analysis and Prediction Platform** built using Python, Machine Learning, Streamlit, Pandas, and Plotly.

The platform helps HR professionals and business teams understand employee attrition patterns, explore workforce data, identify factors associated with employee turnover, and predict employee attrition using a Logistic Regression model.

## 🚀 Live Demo

**[Click Here to View Employee Attrition Intelligence](https://employee-attrition-rate-uifqwjbyjnkthlujb7vmhu.streamlit.app/)**

## 💻 GitHub Repository

[Employee Attrition Rate – GitHub](https://github.com/shreya975/Employee-Attrition-Rate)

---

## 📌 Project Overview

Employee attrition is a major challenge for organizations because employee turnover can affect productivity, workforce stability, and recruitment costs.

This project uses employee data and machine learning techniques to analyze attrition patterns and provide meaningful workforce insights.

It combines interactive dashboards, statistical analysis, employee-level exploration, and predictive modeling in a single Streamlit application.

## ✨ Key Features

### 1. Overview Dashboard
- Overall workforce overview.
- Employee count and attrition KPIs.
- Interactive charts and visualizations.
- Summary of important workforce metrics.

### 2. Workforce Analytics
- Employee distribution by department.
- Age and income analysis.
- Job role and experience analysis.
- Workforce demographics and satisfaction analysis.

### 3. Attrition Analysis
- Attrition rate analysis.
- Department-wise attrition patterns.
- Job role-based attrition analysis.
- Overtime and employee satisfaction analysis.
- Identification of important attrition trends.

### 4. Employee Explorer
- Explore individual employee records.
- Filter employees using different attributes.
- Analyze employee characteristics and workforce patterns.

### 5. Predict Attrition
- Machine Learning-based employee attrition prediction.
- Logistic Regression classification model.
- Employee information-based predictions.
- Prediction probability using model output.

### 6. Model Performance
- Held-out test dataset evaluation.
- Accuracy, Precision, Recall, and F1-score.
- ROC-AUC analysis.
- Confusion Matrix.
- ROC Curve.
- Model coefficient analysis.

### 7. Business Insights
- Understand important attrition factors.
- Identify workforce trends.
- Support data-driven HR decisions.

---

## 📈 Machine Learning Model Performance

The Logistic Regression model was evaluated using a separate held-out test dataset containing 294 employee records.

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 88.78% |
| Precision | 76.92% |
| Recall | 42.55% |
| F1-Score | 54.79% |
| ROC-AUC | 0.8296 |

*These results represent the reported held-out evaluation, not the in-sample sanity check.*

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Plotly, Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Model Serialization | Joblib |
| Frontend / Dashboard | Streamlit |
| Development Environment | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |
| Deployment | Streamlit Community Cloud |

## 📂 Project Structure

```text
Employee-Attrition-Rate/
│
├── app.py
├── requirements.txt
│
├── HR-Employee-Attrition.csv
├── employee_attrition_model.pkl
├── heldout_test_data.csv
│
├── Employee_Attrition_Analysis_COMPLETE.ipynb
│
└── README.md
```

*The notebook and data files should be included in the repository if you choose to publish them.*

## ⚙️ Installation and Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/shreya975/Employee-Attrition-Rate.git
```

### 2. Navigate to the Project Directory

```bash
cd Employee-Attrition-Rate
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Mac / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## 📊 Dataset

The project uses the IBM HR Employee Attrition dataset, containing employee information such as:

- Age and Gender
- Department and Job Role
- Monthly Income
- Business Travel
- Job Satisfaction
- Work-Life Balance
- Years at Company
- Years in Current Role
- Overtime
- Attrition Status

The dataset supports workforce analysis, exploratory data analysis, and machine learning-based attrition prediction.

## 🎯 Project Objectives

- Analyze employee attrition patterns using data analytics.
- Identify workforce characteristics associated with employee turnover.
- Build interactive HR analytics dashboards.
- Apply machine learning for employee attrition prediction.
- Evaluate classification performance using appropriate metrics.
- Present actionable workforce insights through data visualization.

## 🔮 Future Improvements

- Add advanced machine learning models for comparison.
- Improve attrition prediction recall.
- Introduce explainable AI using SHAP.
- Add employee risk segmentation.
- Develop more detailed HR recommendations.
- Integrate real-time workforce data.

## 👩‍💻 Author

**Shreya Mahajan**

Data Science | Data Analytics | Machine Learning

GitHub: [shreya975](https://github.com/shreya975)

---

⭐ If you find this project useful, consider giving the repository a star!
