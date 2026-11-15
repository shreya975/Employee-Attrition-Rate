# 🧠 Employee Attrition Intelligence Platform
### Predictive HR Analytics | Machine Learning | Statistical Analysis | Python | Streamlit

An interactive **Employee Attrition Intelligence Platform** built using Python, Machine Learning, and Data Analytics to understand employee turnover and predict attrition risk. The project combines Exploratory Data Analysis (EDA), statistical analysis, Logistic Regression, predictive modeling, and interactive visualizations to transform employee data into meaningful workforce insights.

<p align="center">
  <a href="https://employee-attrition-rate-j33mghqrbaac9kce8uerfz.streamlit.app/">
    <img src="https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  </a>
  <a href="https://github.com/shreya975/Employee-Attrition-Rate">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white" />
  </a>
</p>

---

## 🌐 Project Links

| Resource | Link |
|---|---|
| Live Application | [Employee Attrition Dashboard](https://employee-attrition-rate-j33mghqrbaac9kce8uerfz.streamlit.app/) |
| GitHub Repository | [Employee-Attrition-Rate](https://github.com/shreya975/Employee-Attrition-Rate) |

---

## 📌 Project Overview

Employee attrition is a major challenge for organizations. Losing experienced employees can increase recruitment costs, affect productivity, disrupt business operations, and result in the loss of valuable organizational knowledge.

Traditional HR reporting generally focuses on understanding employee turnover after it happens. This project takes a predictive analytics approach by analyzing historical employee information to identify patterns associated with attrition.

The platform uses the IBM HR Analytics Employee Attrition & Performance dataset, containing employee demographics, compensation, job roles, satisfaction levels, career progression, and workplace-related information.

A **Logistic Regression classification model** is developed to estimate employee attrition probability. The project also explores workforce characteristics, identifies relationships between employee attributes, and compares standard and class-balanced Logistic Regression models.

The objective is to demonstrate how Data Analytics and Machine Learning can support HR decision-making and employee retention planning.

### 🎯 Project Objectives

- Predict employee attrition using supervised Machine Learning.
- Identify patterns associated with employee turnover.
- Analyze department-wise employee attrition.
- Understand relationships between satisfaction and retention.
- Explore salary and compensation-related workforce patterns.
- Investigate the relationship between overtime and attrition.
- Analyze employee experience and career progression.
- Generate employee-level attrition predictions.
- Evaluate classification performance using multiple metrics.
- Support data-driven workforce planning through analytical insights.

---

## 📊 Dataset Overview

**Dataset:** IBM HR Analytics Employee Attrition & Performance

The project uses structured HR data containing employee demographics, professional information, salary details, satisfaction ratings, and career progression.

### Dataset Statistics

| Metric | Value |
|---|---:|
| Total Employee Records | 1,470 |
| Original Dataset Columns | 35 |
| Employees Who Stayed | 1,233 |
| Employees Who Left | 237 |
| Attrition Rate | 16.12% |
| Retention Rate | 83.88% |
| Missing Values | 0 |
| Original Target Variable | Attrition |
| Machine Learning Problem | Binary Classification |

### Employee Attrition Distribution

| Employee Status | Count | Percentage |
|---|---:|---:|
| Stayed (No) | 1,233 | 83.88% |
| Left (Yes) | 237 | 16.12% |
| **Total** | **1,470** | **100%** |

### Dataset Feature Categories

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

### Target Variable

`Attrition`

- `0` = Employee stayed with the organization.
- `1` = Employee left the organization.

---

## 🔍 Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand workforce characteristics, identify patterns, and discover relationships between HR variables.

### 1. Dataset Inspection

- Dataset dimensions and structure.
- Data types and feature identification.
- Missing value analysis.
- Unique value analysis.
- Descriptive statistical summary.
- Numerical and categorical feature analysis.

### 2. Univariate Analysis

- Employee age distribution.
- Attrition count distribution.
- Attrition percentage distribution.
- Numerical feature distributions.
- Employee compensation patterns.

### 3. Bivariate Analysis

- Age vs Attrition.
- Department vs Attrition.
- Job Role vs Attrition.
- Overtime vs Attrition.
- Monthly Income vs Attrition.
- Job Satisfaction vs Attrition.
- Work-Life Balance vs Attrition.
- Years at Company vs Attrition.
- Gender vs Attrition.

### 4. Statistical Analysis

- Mean and standard deviation.
- Minimum and maximum values.
- Quartile analysis.
- Group-wise average comparisons.
- Correlation analysis.
- Feature relationship analysis.

### 5. Correlation Analysis

A correlation heatmap was generated to understand relationships between numerical variables.

The strongest negative correlations with attrition in the original dataset included:

| Feature | Correlation |
|---|---:|
| Total Working Years | -0.1711 |
| Job Level | -0.1691 |
| Years in Current Role | -0.1605 |
| Monthly Income | -0.1598 |
| Age | -0.1592 |
| Years With Current Manager | -0.1562 |
| Stock Option Level | -0.1371 |
| Years at Company | -0.1344 |
| Job Involvement | -0.1300 |
| Job Satisfaction | -0.1035 |

**Interpretation:** These correlations show associations within the dataset. They do not establish that any individual factor causes employee attrition.

### 6. Department-Wise Attrition Analysis

The following attrition rates were calculated from the dataset:

| Department | Attrition Rate |
|---|---:|
| Sales | 20.63% |
| Human Resources | 19.05% |
| Research & Development | 13.84% |

Sales has the highest observed attrition rate among the three departments in this dataset.

---

## 🧹 Data Cleaning & Preprocessing

The original dataset contains 1,470 records and 35 columns.

The preprocessing workflow prepares the data for Machine Learning while preventing target leakage.

### Preprocessing Steps

- Loaded the original HR dataset using Pandas.
- Inspected dataset dimensions and data types.
- Checked missing values.
- Encoded the target variable into binary values.
- Removed irrelevant identifier and constant columns.
- Separated independent and dependent variables.
- Identified numerical and categorical features.
- Applied stratified train-test splitting.
- Standardized numerical variables.
- Encoded categorical variables using One-Hot Encoding.

### Feature Selection

The following four columns were removed:

- EmployeeNumber
- EmployeeCount
- Over18
- StandardHours

These columns were excluded because they represent an employee identifier or constant information.

| Feature Summary | Count |
|---|---:|
| Original Columns | 35 |
| Removed Columns | 4 |
| Remaining Columns | 31 |
| Independent Features | 30 |
| Target Variable | 1 |

### Numerical Feature Scaling

**StandardScaler** was used to standardize numerical features.

Standardization transforms numerical variables to a common scale using their mean and standard deviation.

### Categorical Feature Encoding

**OneHotEncoder** was used to convert categorical variables into numerical representations.

The encoder uses `handle_unknown="ignore"` to handle categories that were not encountered during training.

### Data Leakage Prevention

The preprocessing workflow was implemented using Scikit-learn's `ColumnTransformer` and `Pipeline`.

This ensures that:
- The training data is used to fit preprocessing transformations.
- The testing data is transformed without fitting on it.
- Numerical scaling and categorical encoding remain consistent.
- The target variable is excluded from feature preprocessing.

---

## 🤖 Machine Learning Model

### Logistic Regression

The primary Machine Learning algorithm used in this project is Logistic Regression.

It is a supervised classification algorithm that estimates the probability of an employee leaving or staying based on employee-related characteristics.

### Model Configuration

| Parameter | Configuration |
|---|---|
| Algorithm | Logistic Regression |
| Learning Type | Supervised Learning |
| Problem Type | Binary Classification |
| Input Features | 30 |
| Training Records | 1,176 |
| Testing Records | 294 |
| Train-Test Ratio | 80:20 |
| Random State | 10 |
| Numerical Scaling | StandardScaler |
| Categorical Encoding | OneHotEncoder |
| Maximum Iterations | 1,000 |

### Train-Test Distribution

The dataset was divided using `train_test_split()` with stratification.

| Dataset | Total Records | Stayed | Left |
|---|---:|---:|---:|
| Training | 1,176 | 986 | 190 |
| Testing | 294 | 247 | 47 |
| **Total** | **1,470** | **1,233** | **237** |

Stratified splitting maintains approximately the same class distribution in both training and testing datasets.

---

## ⚖️ Handling Class Imbalance

The original dataset is imbalanced because approximately 83.88% of employees stayed while 16.12% left.

To understand the effect of class imbalance, two Logistic Regression configurations were evaluated.

### Model 1: Standard Logistic Regression

- Default class weighting.
- Focuses on overall classification performance.
- Uses the default probability classification threshold.

### Model 2: Balanced Logistic Regression

- Uses `class_weight="balanced"`.
- Automatically adjusts class weights according to class frequencies.
- Gives additional importance to the minority attrition class.
- Attempts to improve identification of employees who leave.

Both models were evaluated using the same training and testing split.

---

## 📈 Model Performance & Evaluation

The project compares two Logistic Regression configurations using a held-out testing dataset of 294 employee records.

### Model Comparison

| Evaluation Metric | Logistic Regression | Balanced Logistic Regression |
|---|---:|---:|
| Accuracy | 88.78% | 78.23% |
| Precision | 76.92% | 39.76% |
| Recall | 42.55% | 70.21% |
| F1-Score | 54.79% | 50.77% |
| ROC-AUC | 0.8296 | 0.8240 |

### Selected Model: Logistic Regression

The standard Logistic Regression model was selected because it achieved a higher mean F1-score during 5-fold cross-validation on the training dataset.

The balanced model achieved higher recall, but its precision and overall F1-score were lower.

### Final Model Results

| Metric | Final Score |
|---|---:|
| Accuracy | 88.78% |
| Precision | 76.92% |
| Recall | 42.55% |
| F1-Score | 54.79% |
| ROC-AUC | 0.8296 |

### Confusion Matrix

The final Logistic Regression model produced the following confusion matrix:

| Actual / Predicted | Stayed | Left |
|---|---:|---:|
| Stayed | 241 | 6 |
| Left | 27 | 20 |

### Confusion Matrix Interpretation

- **241 True Negatives:** Employees who stayed and were correctly predicted to stay.
- **20 True Positives:** Employees who left and were correctly identified.
- **6 False Positives:** Employees predicted to leave who actually stayed.
- **27 False Negatives:** Employees who left but were predicted to stay.

### Classification Performance Interpretation

**Accuracy – 88.78%**

The model correctly classified approximately 88.78% of all testing records.

**Precision – 76.92%**

Among employees predicted to leave, approximately 76.92% actually left.

**Recall – 42.55%**

The model identified approximately 42.55% of employees who actually left.

**F1-Score – 54.79%**

The F1-score represents the balance between precision and recall.

**ROC-AUC – 0.8296**

The ROC-AUC measures how well the model distinguishes between employees who stayed and employees who left across different classification thresholds.

### Important Observation

Although the standard Logistic Regression model achieved 88.78% accuracy, its recall for the attrition class is relatively low.

The balanced model improves recall to 70.21%, but increases false-positive predictions.

This highlights the importance of evaluating multiple classification metrics rather than relying only on accuracy.

---

## 🔄 Cross-Validation

To assess model consistency, 5-fold Stratified Cross-Validation was performed using the training dataset.

### Cross-Validation Configuration

| Parameter | Value |
|---|---|
| Technique | Stratified K-Fold |
| Number of Folds | 5 |
| Evaluation Metric | F1-Score |
| Random State | 10 |

### Cross-Validation Results

| Fold | Logistic Regression | Balanced Logistic Regression |
|---|---:|---:|
| 1 | 0.5484 | 0.5043 |
| 2 | 0.5373 | 0.4463 |
| 3 | 0.4615 | 0.4821 |
| 4 | 0.5085 | 0.5660 |
| 5 | 0.6441 | 0.5686 |
| **Mean F1-Score** | **0.5400** | **0.5135** |

The standard Logistic Regression model achieved a higher mean cross-validation F1-score and was selected as the final model.

---

## 📊 Feature Importance & Model Interpretation

Logistic Regression coefficients were analyzed to understand how transformed employee features contribute to the model's attrition predictions.

The top features by absolute coefficient magnitude in the trained model included:

| Feature | Coefficient |
|---|---:|
| OverTime_No | -1.0173 |
| OverTime_Yes | 0.9940 |
| BusinessTravel_Non-Travel | -0.9218 |
| BusinessTravel_Travel_Frequently | 0.8571 |
| JobRole_Sales Representative | 0.7456 |
| JobRole_Research Director | -0.7152 |
| JobRole_Laboratory Technician | 0.6959 |
| MaritalStatus_Single | 0.6437 |
| YearsSinceLastPromotion | 0.5892 |
| EducationField_Human Resources | 0.5877 |

### Interpretation

- Positive coefficients are associated with higher predicted attrition odds.
- Negative coefficients are associated with lower predicted attrition odds.
- Larger absolute coefficients indicate stronger model contributions on the transformed feature scale.

These coefficients represent model associations and should not be interpreted as causal relationships.

---

## 📉 Visualizations & Analytical Outputs

The project includes multiple visualizations to communicate employee patterns and Machine Learning results.

### Exploratory Data Analysis
- Employee attrition distribution.
- Attrition percentage pie chart.
- Age distribution histogram.
- Age vs attrition boxplot.
- Department-wise attrition count.
- Job role-wise attrition distribution.
- Overtime vs attrition.
- Monthly income vs attrition.
- Job satisfaction analysis.
- Work-life balance analysis.
- Years at company vs attrition.
- Gender-wise attrition.

### Statistical Analysis
- Descriptive statistical summary.
- Numerical feature correlation heatmap.
- Top features correlated with attrition.
- Department-wise attrition rate.
- Group-wise employee metrics.

### Machine Learning Evaluation
- Confusion matrix.
- ROC curve.
- Actual vs predicted comparison.
- Logistic Regression model comparison.
- 5-fold cross-validation visualization.
- Logistic Regression coefficient analysis.

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
             Statistical Analysis
                      |
                      ▼
             Correlation Analysis
                      |
                      ▼
              Feature Selection
                      |
                      ▼
             Target Encoding
                      |
                      ▼
             Train-Test Split
                      |
                      ▼
           Numerical Standardization
                      |
                      ▼
              One-Hot Encoding
                      |
                      ▼
           Logistic Regression Model
                      |
                      ▼
             Model Evaluation
                      |
                      ▼
          Class Imbalance Comparison
                      |
                      ▼
             Cross-Validation
                      |
                      ▼
           Feature Interpretation
                      |
                      ▼
              Final Model
                      |
                      ▼
             Model Serialization
                      |
                      ▼
           Streamlit Deployment
```

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn, Plotly |
| Statistical Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Classification Algorithm | Logistic Regression |
| Numerical Scaling | StandardScaler |
| Categorical Encoding | OneHotEncoder |
| Data Preprocessing | ColumnTransformer |
| ML Workflow | Scikit-learn Pipeline |
| Model Evaluation | Scikit-learn Metrics |
| Model Serialization | Joblib |
| Development Environment | Jupyter Notebook |
| Application Framework | Streamlit |
| Version Control | Git, GitHub |
| Deployment | Streamlit Community Cloud |

---

## 📂 Project Structure

```text
Employee-Attrition-Rate/
│
├── Employee_Attrition_Analysis.ipynb
│   └── Complete EDA, statistical analysis,
│       preprocessing, model training and evaluation
│
├── HR-Employee-Attrition.csv
│   └── IBM HR Analytics dataset
│
├── app.py
│   └── Streamlit application
│
├── employee_attrition_model.pkl
│   └── Serialized final Machine Learning pipeline
│
├── model_evaluation_results.csv
│   └── Final model evaluation metrics
│
├── requirements.txt
│   └── Project dependencies
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
pip install pandas numpy matplotlib seaborn scikit-learn streamlit plotly joblib
```

Alternatively, install the dependencies from the requirements file:

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 💼 Business Applications

The project demonstrates how Machine Learning and HR analytics can support workforce decision-making.

Potential applications include:

- Understanding employee turnover patterns.
- Identifying employee groups that may need further investigation.
- Supporting employee retention planning.
- Evaluating satisfaction and compensation trends.
- Improving workforce reporting.
- Supporting HR resource planning.
- Making workforce analytics more accessible through interactive applications.

**Note:** Model predictions are analytical estimates, not definitive judgments about employees. They should not be used as the sole basis for employment decisions.

---

## 🔮 Future Enhancements

Potential improvements to expand the project:

- Compare Logistic Regression with Random Forest and XGBoost.
- Implement hyperparameter tuning.
- Add advanced cross-validation strategies.
- Experiment with different probability thresholds.
- Implement SHAP-based explainable AI.
- Improve attrition-class recall while maintaining useful precision.
- Add employee-level prediction explanations.
- Develop an advanced HR recommendation engine.
- Integrate FastAPI for model serving.
- Add database integration for employee information.
- Implement automated model monitoring and retraining.
- Expand workforce analytics and reporting capabilities.

---

## 👩‍💻 Author

**Shreya Mahajan**

Aspiring Data Scientist | Data Analytics | Machine Learning | Python

- **GitHub:** [shreya975](https://github.com/shreya975)
- **LinkedIn:** [Connect on LinkedIn](https://www.linkedin.com/in/shreya-mahajan-b38b28385/)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

<p align="center">
  ⭐ If you find this project interesting, consider giving the repository a star!
</p>

<p align="center">
  **Turning HR Data into Meaningful Workforce Insights.**
</p>
