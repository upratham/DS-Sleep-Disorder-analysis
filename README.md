# Sleep Disorder Prediction (ML Classification)

Predict whether a person has a sleep disorder using lifestyle and basic health indicators (sleep duration/quality, stress, activity, BMI, blood pressure, etc.).  
This project explores data cleaning, EDA, feature engineering, class imbalance handling, and compares multiple ML models.

---

## Project Goal

**Research Question:**  
Can we predict whether a person has a sleep disorder from lifestyle and other health records?

Why it matters: sleep disorders affect quality of life and may lead to long-term health issues. Early risk detection can help people make lifestyle changes sooner.

---

## Dataset

This project uses the Sleep Health & Lifestyle dataset (including an extended version).

- Kaggle (extended): https://www.kaggle.com/datasets/fortressenebe/sleep-health-and-lifestyle-dataset-extended/data  
- Kaggle (original): https://www.kaggle.com/datasets/henryshan/sleep-health-and-lifestyle

### Key Columns (13 total)

- `Gender`, `Age`, `Occupation`
- `Sleep Duration`, `Quality of Sleep`
- `Physical Activity Level`, `Stress Level`
- `BMI Category`, `Blood Pressure`
- `Heart Rate`, `Daily Steps`
- **Target:** `Sleep Disorder`

### Target Classes (example distribution from notebook)

- No Disorder: 946  
- Sleep Apnea: 175  
- Insomnia: 167  
- Restless Leg Syndrome: 102  
- Narcolepsy: 87  

> Note: The notebook also handles missing values and type inconsistencies in the `Sleep Disorder` column.

---

## Methods & Workflow

The notebook follows a typical ML pipeline:

1. **Data collection & loading**
2. **Cleaning & preprocessing**
   - Handling missing values (notably in target column)
   - Encoding categorical features (OneHot/Ordinal)
   - Scaling numeric features (MinMaxScaler)
3. **Exploratory Data Analysis (EDA)**
   - Relationship exploration between lifestyle indicators and sleep disorder
   - Outlier checks (e.g., Z-score)
4. **Modeling**
   - Baseline model(s)
   - **AdaBoost**
   - **Random Forest**
   - Neural Network (ANN) using Keras/TensorFlow
5. **Class imbalance handling**
   - SMOTE oversampling used for training data
6. **Evaluation**
   - Accuracy, classification report, confusion matrix
   - Model comparison chart

---

## Results Snapshot (from notebook runs)

- Baseline test accuracy: ~0.625  
- **AdaBoost accuracy:** ~0.642 (best among compared models in the notebook)
- Random Forest accuracy: ~0.622  
- ANN accuracy: ~0.598  

> These are dataset- and split-dependent; reruns may vary.

---

## Repository Contents

- `Sleep_Disorder_analysis.ipynb` — main notebook with the full pipeline

Suggested structure (optional but recommended):
```text
.
├── Sleep_Disorder_analysis.ipynb
├── data/
│   └── ss.csv
├── requirements.txt
└── README.md
