# 🧠 Job Career Level Classification using NLP & Structured Data
A machine learning pipeline that classifies job listings into career levels (e.g. Entry Level, Director, Specialist) using a mix of text features and structured categorical data. The model is built using scikit-learn pipelines, TfidfVectorizer, and RandomForestClassifier.

# 🚀 Project Overview
This project aims to predict the career level of job postings based on features like:
Title: finalproject
Description: Job Career Level Classification using NLP
I  handle a multi-class classification task with imbalanced data, and explore ways to enhance performance using:
✅ Preprocessing Pipelines
✅ Text Vectorization (TF-IDF)
✅ One-Hot Encoding for Categorical Features
✅ Feature Selection (SelectKBest)
✅ Optional Oversampling (SMOTE) for rare classes

# 🧾 Dataset
Format: .ods file (OpenDocument Spreadsheet)
Loaded via pandas.read_excel()
Includes both textual and categorical data
Target column: career_level

# 🔄 Preprocessing
Text Cleaning: Strip state code from location strings using RegEx
Missing Values: Fill missing job descriptions with "missing"
Train-Test Split: 80/20 split, stratified by class labels
Optional SMOTE: Balance minority classes (commented in code)
 Preprocessing Includes:
TfidfVectorizer for: Job Title, Description, Industry
OneHotEncoder for: Location, Function

# Model Evaluation
Overall Accuracy: 76%  
Macro Avg F1-score: 0.39  
Weighted Avg F1-score: 0.74

# Classification report
![Image](https://github.com/user-attachments/assets/2f4e3866-916f-4245-a188-53dc12329ecc)
