# 🧠 Employee Attrition Intelligence Platform
### Predictive HR Analytics | Machine Learning | Workforce Intelligence | Python | Streamlit

An interactive **Employee Attrition Intelligence Platform** that leverages Machine Learning, statistical analysis, and interactive visualizations to understand employee turnover and predict attrition risk. The platform transforms HR data into actionable insights, helping organizations understand workforce patterns, identify potential retention challenges, and support data-driven HR decisions.

<p align="center">
  <a href="https://employee-attrition-rate-j33mghqrbaac9kce8uerfz.streamlit.app/">
    <img src="https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  </a>
  <a href="https://github.com/shreya975/Employee-Attrition-Rate">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white" />
  </a>
</p>

---

## 🌐 Live Application

Explore the interactive Employee Attrition Intelligence Platform:

**🚀 Live Dashboard:** [Open Application](https://employee-attrition-rate-j33mghqrbaac9kce8uerfz.streamlit.app/)

**💻 GitHub Repository:** [View Source Code](https://github.com/shreya975/Employee-Attrition-Rate)

---

## 📌 Project Overview

Employee attrition is one of the major challenges organizations face. Losing employees can increase recruitment expenses, affect productivity, disrupt business operations, and result in the loss of valuable organizational knowledge.

Traditional HR reporting mainly focuses on understanding employee turnover after it happens. This project takes a predictive analytics approach by using historical employee information to identify patterns associated with employee attrition.

The platform analyzes employee demographics, compensation, job satisfaction, work environment, career development, and other organizational factors.

Using a **Logistic Regression Machine Learning model**, the application estimates employee attrition probability and presents workforce insights through an interactive Streamlit dashboard.

The objective is to transform raw HR data into meaningful business intelligence that can help HR professionals understand workforce behavior and make informed retention decisions.

### 🎯 Project Objectives

- Predict employee attrition using supervised Machine Learning.
- Identify patterns and factors associated with employee turnover.
- Analyze department-wise workforce distribution and attrition.
- Understand relationships between employee satisfaction and retention.
- Explore compensation and salary-related workforce patterns.
- Evaluate the influence of overtime and work-life balance.
- Generate employee-level attrition predictions.
- Provide actionable HR insights through interactive visualizations.

---

## 📊 Dataset Overview

**Dataset:** IBM HR Analytics Employee Attrition & Performance Dataset

The project uses structured employee information containing demographic details, job characteristics, compensation, employee satisfaction, and career progression.

### Dataset Statistics

| Metric | Value |
|---|---:|
| Total Employee Records | 1,470 |
| Original Dataset Features | 35 |
| Employees Who Left | 237 |
| Employees Who Stayed | 1,233 |
| Missing Values | 0 |
| Attrition Rate | 16.12% |
| Employee Retention Rate | 83.88% |
| Target Variable | Attrition |
| Problem Type | Binary Classification |

### Employee Attrition Distribution

| Employee Status | Count | Percentage |
|---|---:|---:|
| Attrition – Yes | 237 | 16.12% |
| Attrition – No | 1,233 | 83.88% |
| **Total** | **1,470** | **100%** |

### Dataset Features

The dataset includes the following categories:

**1. Employee Demographics**
- Age
- Gender
- Marital Status
- Education
- Education Field
- Distance From Home

**2. Job and Organizational Information**
- Department
- Job Role
- Job Level
- Business Travel
- Job Involvement
- Job Satisfaction

**3. Compensation and Salary**
- Daily Rate
- Hourly Rate
- Monthly Income
- Monthly Rate
- Percent Salary Hike
- Stock Option Level

**4. Employee Satisfaction**
- Environment Satisfaction
- Job Satisfaction
- Relationship Satisfaction
- Work-Life Balance

**5. Career Development and Experience**
- Total Working Years
- Years at Company
- Years in Current Role
- Years Since Last Promotion
- Years With Current Manager
- Number of Companies Worked
- Training Times Last Year

**6. Performance and Working Conditions**
- Performance Rating
- OverTime
- Employee Attrition

**Target Variable: `Attrition`**

- Yes (1): Employee left the organization.
- No (0): Employee stayed with the organization.

---

## 🔍 Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand employee characteristics, identify patterns, and explore relationships between HR variables.

### Analysis Performed

- Dataset structure and data type inspection.
- Missing value analysis.
- Statistical summary using descriptive statistics.
- Employee attrition distribution.
- Numerical feature correlation analysis.
- Correlation heatmap visualization.
- Feature relationship analysis.
- Identification of highly correlated features.

### Key Business Questions

1. Which departments experience higher employee attrition?
2. Is overtime associated with employee turnover?
3. How does job satisfaction relate to employee retention?
4. Does monthly income show patterns associated with attrition?
5. How does employee experience relate to workforce stability?
6. Does work-life balance have a relationship with employee turnover?
7. Which employee characteristics are useful for attrition prediction?

The EDA process helps convert employee-level information into meaningful workforce insights.

---

## 🧹 Data Cleaning & Preprocessing

Data preprocessing was performed to prepare the dataset for Machine Learning.

### Preprocessing Steps

- Loaded the HR dataset using Pandas.
- Inspected dataset dimensions and data types.
- Checked missing values across all columns.
- Converted the target variable into numerical representation.
- Performed numerical feature standardization.
- Identified highly correlated numerical features.
- Removed selected highly correlated features.
- Applied One-Hot Encoding to categorical variables.
- Divided the dataset into training and testing subsets.

### Feature Selection

The following five columns were removed during the feature selection process because of high correlation with other variables:

- JobLevel
- PerformanceRating
- YearsWithCurrManager
- YearsInCurrentRole
- TotalWorkingYears

| Feature Summary | Count |
|---|---:|
| Original Columns | 35 |
| Removed Columns | 5 |
| Remaining Columns | 30 |
| Input Features | 29 |
| Target Variable | 1 |

### Feature Scaling

**StandardScaler** was used to standardize numerical variables.

Standardization transforms numerical values to a common scale using the mean and standard deviation.

### Categorical Encoding

**One-Hot Encoding** was applied to categorical features to convert categorical information into a numerical format suitable for the Logistic Regression model.

---

## 🤖 Machine Learning Model

### Logistic Regression

The project uses Logistic Regression, a supervised Machine Learning classification algorithm.

It estimates the probability of employee attrition based on employee-related characteristics.

### Model Training Configuration

| Parameter | Value |
|---|---|
| Algorithm | Logistic Regression |
| Learning Type | Supervised Learning |
| Problem Type | Binary Classification |
| Training Samples | 1,176 |
| Testing Samples | 294 |
| Train-Test Split | 80:20 |
| Random State | 10 |
| Numerical Scaling | StandardScaler |
| Categorical Encoding | One-Hot Encoding |
| Maximum Iterations | 1,000 |

### Model Workflow

1. Prepare employee data.
2. Separate independent and dependent variables.
3. Perform an 80:20 train-test split.
4. Apply One-Hot Encoding to categorical variables.
5. Train the Logistic Regression model.
6. Generate predictions on testing data.
7. Evaluate classification performance.
8. Visualize the confusion matrix.

---

## 📈 Model Performance & Evaluation

The original Logistic Regression experiment achieved an accuracy of **88.44%** on the test dataset.

### Evaluation Metrics

| Metric | Score |
|---|---:|
| Accuracy | 88.44% |
| Precision (Attrition) | 86.21% |
| Recall (Attrition) | 45.45% |
| F1-Score (Attrition) | 59.52% |

### Confusion Matrix

| Actual / Predicted | Stayed (0) | Left (1*) |
|---|---:|---:|
| Stayed | 235 | 4 |
| Left | 30 | 25 |

*The original notebook internally produces class labels 0 and 2 because the target was standardized before conversion to integers. The labels above are shown using their business meanings.*

### Performance Interpretation

- **235 True Negatives:** Employees who stayed and were correctly classified.
- **25 True Positives:** Employees who left and were correctly identified.
- **4 False Positives:** Employees predicted to leave but who stayed.
- **30 False Negatives:** Employees who left but were predicted to stay.

### Important Model Limitation

The original notebook applies StandardScaler to all numerical columns, including the target variable, before the train-test split. This introduces data leakage and changes the original target labels.

Therefore, the reported metrics represent the original notebook experiment and should be treated as preliminary results, not validated production performance.

A corrected preprocessing pipeline should exclude the target from scaling, fit preprocessing only on training data, and retrain the model before final performance claims.

---

## 📊 Interactive HR Analytics & Dashboard

The Streamlit application provides an interactive interface for exploring employee information and attrition predictions.

### Core Analytical Capabilities

**1. Workforce Analytics**
- Employee workforce overview.
- Attrition and retention distribution.
- Workforce stability analysis.
- Employee-level information.

**2. Department-Wise Analysis**
- Department-level attrition comparisons.
- Workforce distribution.
- Department-specific insights.

**3. Compensation Analysis**
- Monthly income analysis.
- Salary distribution.
- Compensation-related workforce patterns.

**4. Employee Satisfaction Analysis**
- Job satisfaction.
- Environment satisfaction.
- Relationship satisfaction.
- Work-life balance.

**5. Feature Analysis**
- Model coefficient-based feature analysis.
- Visualization of influential features.
- Understanding relationships between employee characteristics and predictions.

**6. Employee Attrition Prediction**
- Individual employee information input.
- Predicted attrition status.
- Attrition probability.
- Employee risk assessment.

**7. HR Recommendations**
- Overtime-related workload review.
- Employee satisfaction improvement.
- Compensation analysis.
- Career progression considerations.
- Work-life balance improvement.

---

## 🔄 End-to-End Project Workflow

```text
          HR Analytics Dataset
                  |
                  ▼
          Data Collection
                  |
                  ▼
          Data Inspection
                  |
                  ▼
       Data Cleaning & EDA
                  |
                  ▼
        Correlation Analysis
                  |
                  ▼
          Feature Selection
                  |
                  ▼
         Data Preprocessing
                  |
                  ▼
         Train-Test Split
                  |
                  ▼
         One-Hot Encoding
                  |
                  ▼
       Logistic Regression
                  |
                  ▼
        Model Evaluation
                  |
                  ▼
       Attrition Prediction
                  |
                  ▼
        Employee Risk Analysis
                  |
                  ▼
       Interactive HR Dashboard
                  |
                  ▼
         Business Insights
```

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn |
| ML Algorithm | Logistic Regression |
| Feature Encoding | Category Encoders |
| Feature Scaling | StandardScaler |
| Application Framework | Streamlit |
| Development Environment | Jupyter Notebook |
| Version Control | Git, GitHub |
| Deployment | Streamlit Community Cloud |

---

## 📂 Project Structure

```text
Employee-Attrition-Rate/
│
├── Employee Attrition Rate.ipynb
│   └── Data analysis and machine learning workflow
│
├── HR-Employee-Attrition.csv
│   └── Employee HR analytics dataset
│
├── app.py
│   └── Streamlit application
│
├── requirements.txt
│   └── Python dependencies
│
├── LICENSE
│   └── Project license
│
└── README.md
    └── Project documentation
```

---

## 🚀 Installation & Setup

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

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 📦 Python Libraries

The project uses the following core libraries:

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Category Encoders
- Plotly
- Streamlit

Install the dependencies using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn category_encoders plotly streamlit
```

---

## 💼 Business Applications

This platform demonstrates how Machine Learning and HR analytics can support organizational decision-making.

Potential business applications include:

- Understanding employee turnover patterns.
- Identifying workforce segments requiring further investigation.
- Supporting employee retention planning.
- Evaluating employee satisfaction trends.
- Analyzing compensation-related patterns.
- Supporting workforce planning and HR reporting.
- Making HR analytics more accessible through interactive dashboards.

**Note:** Predictions are analytical indicators, not definitive judgments about individual employees. They should not be used as the sole basis for employment decisions.

---

## 🔮 Future Enhancements

Potential improvements to make the platform more advanced:

- Compare Logistic Regression with Random Forest and XGBoost.
- Implement cross-validation and hyperparameter tuning.
- Add SHAP-based explainable AI.
- Improve model evaluation using ROC-AUC and classification reports.
- Develop an advanced employee retention recommendation engine.
- Add automated model retraining.
- Integrate a FastAPI prediction service.
- Add database integration for employee information.
- Develop advanced workforce analytics and reporting.

---

## 👩‍💻 Author

**Shreya Mahajan**

Aspiring Data Scientist | Data Analytics | Machine Learning | Python

- **GitHub:** [shreya975](https://github.com/shreya975)
- **LinkedIn:** [Connect on LinkedIn](https://www.linkedin.com/in/shreya-mahajan-b38b28385/)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

---

<p align="center">
  ⭐ If you find this project interesting, consider giving the repository a star!
</p>

<p align="center">
  <b>Turning HR Data into Meaningful Workforce Insights.</b>
</p>
