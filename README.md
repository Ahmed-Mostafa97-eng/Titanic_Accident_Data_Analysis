# Titanic Survival Analysis

This repository contains a comprehensive exploratory data analysis (EDA) of the Titanic passenger dataset. The primary goal of this analysis is to understand the factors that influenced passenger survival rates during the tragic sinking of the Titanic. This project provides a detailed walkthrough of the data cleaning process, feature engineering, and the discovery of key survival patterns through statistical analysis and data visualization.

## Project Overview

The sinking of the Titanic is one of the most infamous shipwrecks in history. While there was an element of luck involved in surviving, it seems some groups of people were more likely to survive than others. This analysis uses passenger data (e.g., age, gender, class) to uncover these patterns and provide a clear, data-driven narrative of the event.

## Questions (KPIs)
- What is the approximate percentage of passengers who survived the disaster?**- Which month got the highest sales and orders?
- Did female passengers have a higher survival rate than male passengers? 
- Which passenger class (Pclass) had the highest survival rate? Which had the lowest?
- Did children (under 18 years old) have a better chance of survival compared to adults?
- Which embarkation port had the highest number of survivors? Are there noticeable differences between the three ports?
- Did passengers traveling with family members (SibSp > 0 or Parch > 0) have a better chance of survival compared to those traveling alone?
- Did passengers who paid higher ticket fares have better survival chances? What could be the possible reason?
- Is there a relationship between having a recorded cabin number and the chance of survival?
- What were the ages of the oldest and youngest survivors?
- What was the difference in survival rates between third-class passengers and passengers from first and second class combined?
- Which passenger class had the highest average age? What might this suggest about the type of travelers in that class?
- If you split passengers into three groups based on fare (Low, Medium, High), which group had the highest survival rate?
- Who were the top 10 passengers who paid the highest fares, and did they survive?
- If you create a “Group Size” column (SibSp + Parch + 1), do larger groups have better or worse survival chances?
- Among female passengers, which class had the highest survival rate? Does the same trend hold for male passengers?

## Dataset

The dataset used in this analysis is the classic Titanic dataset, sourced from the Kaggle competition "Titanic: Machine Learning from Disaster". It contains information about 891 passengers.

*   **Source**: titanic_analysis.py

### Data Dictionary

| Variable    | Definition                                 | Key                                            |
|-------------|--------------------------------------------|------------------------------------------------|
| `PassengerId` | Unique ID for each passenger               |                                                |
| `Survived`  | Survival status                            | 0 = No, 1 = Yes                                |
| `Pclass`    | Ticket class                               | 1 = 1st, 2 = 2nd, 3 = 3rd                      |
| `Sex`       | Gender                                     |                                                |
| `Age`       | Age in years                               |                                                |
| `SibSp`     | # of siblings / spouses aboard the Titanic |                                                |
| `Parch`     | # of parents / children aboard the Titanic |                                                |
| `Ticket`    | Ticket number                              |                                                |
| `Fare`      | Passenger fare                             |                                                |
| `Cabin`     | Cabin number                               |                                                |
| `Embarked`  | Port of Embarkation                        | C = Cherbourg, Q = Queenstown, S = Southampton |

## Exploratory Data Analysis and Methodology

The analysis was conducted using a structured approach, starting from data exploration and cleaning to visualization and insight generation.

### 1. Data Loading and Initial Exploration

The dataset was loaded into a pandas DataFrame. An initial assessment was performed to understand its structure, including the shape, memory usage, data types, and the extent of missing values.

### 2. Data Cleaning and Preprocessing

A rigorous data cleaning process was applied:

*   **Handling Missing Values**:
    *   `Age`: Missing age values were imputed using the median age, grouped by passenger class (`Pclass`) and gender (`Sex`). This provides a more accurate estimation than using a simple overall median.
    *   `Embarked`: The two missing `Embarked` values were filled with the mode of the column (the most frequent port of embarkation).
    *   `Cabin`: The `Cabin` column was dropped from the dataset due to a high percentage of missing values (over 77%), making it unsuitable for reliable analysis.
*   **Data Type Conversion**: Columns like `Pclass`, `Sex`, and `Embarked` were converted to the `category` data type to optimize memory usage and reflect their categorical nature.

### 3. Feature Engineering

To derive deeper insights, several new features were created:

*   `FamilySize`: Calculated by summing `SibSp` and `Parch` and adding 1 (for the passenger themselves).
*   `IsAlone`: A binary feature indicating whether a passenger was traveling alone (`FamilySize` == 1).
*   `Title`: Extracted from the `Name` column (e.g., Mr, Mrs, Miss). Rare titles were grouped into a single 'Rare' category.
*   `AgeGroup`: Passengers were categorized into age groups (Child, Teen, Adult, Middle-aged, Senior).
*   `FareGroup`: Fares were divided into quartiles (Low, Medium, High, Very High).

## Key Findings and Visualizations

The analysis revealed several strong patterns related to survival.

### Comprehensive Analysis Dashboard

The following dashboard summarizes the key findings from the exploratory data analysis:

![Comprehensive Analysis Dashboard](https://private-us-east-1.manuscdn.com/sessionFile/qrtPrkVIY8OqRtct8m7kwW/sandbox/RiKvcmuMAD2bO7y5EY90lP-images_1759245669811_na1fn_L2hvbWUvdWJ1bnR1L3RpdGFuaWMtc3Vydml2YWwtYW5hbHlzaXMvcmVzdWx0cy9maWd1cmVzL3RpdGFuaWNfY29tcHJlaGVuc2l2ZV9hbmFseXNpcw.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvcXJ0UHJrVklZOE9xUnRjdDhtN2t3Vy9zYW5kYm94L1JpS3ZjbXVNQUQyYk83eTVFWTkwbFAtaW1hZ2VzXzE3NTkyNDU2Njk4MTFfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzUnBkR0Z1YVdNdGMzVnlkbWwyWVd3dFlXNWhiSGx6YVhNdmNtVnpkV3gwY3k5bWFXZDFjbVZ6TDNScGRHRnVhV05mWTI5dGNISmxhR1Z1YzJsMlpWOWhibUZzZVhOcGN3LnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc5ODc2MTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=iHJ~By1Q6mB4hPCqDVOVZcSpqD~nl4QYMmoC0wokSoAYivvqoA0xKYDo1e4JutVRsYOb7LUXItVHrw4i83SIP9LZfIuU-3k2uhMvfm4bc2TwCd9IQBX4JDRWc5fMTLaGzU6Cn4zEb1UGWNSoixid55icuBSsSihLnQ~jjDK9T67eiLmfJQ0ZgxqmPiP0QvoN02fPqDSg6Tx40oDjfXdg3pBs85TnrDaGJnm~VQVb8fM-sQJQvmL1odwEUd0jvMmQaeZ-96lrerCGIEZN9j3gjK6fyKtGJw-QpX8oa9e6cUR4FGxLyUCRnCyBF28jKiVDcovU4IeFRuOb~Y4IfjgMRw__)

### Summary of Insights

1.  **Gender was a primary factor**: Women had a significantly higher survival rate (74.2%) compared to men (18.9%). This reflects the "women and children first" protocol.

2.  **Socio-economic status mattered**: 1st class passengers had a 63.0% survival rate, substantially higher than the 24.2% for 3rd class passengers.

3.  **Age played a crucial role**: Children (age < 12) had a high survival rate of 58.0%.

4.  **Family size influenced survival**: Passengers traveling with family had a better chance of survival (50.6%) than those traveling alone (30.4%).

5.  **Fare correlated with survival**: Passengers who paid higher fares, which is a proxy for wealth and class, had a higher survival rate.

## How to Run the Analysis

To replicate this analysis, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    ```

2.  **Set up a virtual environment** (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file should be created for a production-ready repository)*

4.  **Run the analysis script**:
    ```bash
    cd src
    python3 titanic_analysis.py
    ```

    The script will execute the complete analysis pipeline, print insights to the console, and save the visualizations in the `results/figures` directory.

## Repository Structure

```
.titanic-survival-analysis/
├── data/
│   ├── titanic.csv           # Original dataset
│   └── titanic_cleaned.csv   # Cleaned dataset generated by the script
├── notebooks/
│   └── Titanic_Task_Ahmed_Ezzat.ipynb # Original user-provided notebook
├── results/
│   └── figures/
│       └── titanic_comprehensive_analysis.png # Generated visualization dashboard
├── src/
│   └── titanic_analysis.py   # Main analysis script
└── README.md                 # This file
```

