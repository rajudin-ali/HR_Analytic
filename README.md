# HR_Analytic (Ford Sentence Classification Dataset)

## Goals
This project aims to develop a job description analysis model to identify, categorize, and align the required competencies with company needs during the CV screening stage, thereby making the process more efficient and accurate

## Problem Statement

- Traditional CV screening struggles to systematically identify candidate characteristics.

- No automated model existed to classify CV contents into categories such as Responsibility, Skill, Education, and Soft Skills.

- Manual processes were time-consuming and prone to bias

## Methodology

1. Data Preparation – Text cleaning, tokenization, lemmatization, and oversampling (SMOTE).

2. Feature Engineering – Used TF-IDF, Word2Vec, and GloVe vectorization.

3. Model Development & Tuning – Compared SVM, Random Forest, XGBoost, Gradient Boosting, Naïve Bayes, and KNN.

4. Evaluation Metrics – Accuracy, Precision, Recall, F1-score, Confusion Matrix.

5. Deployment – Implemented via Streamlit app for recruiters to analyze CV text.

<img width="376" height="170" alt="image" src="https://github.com/user-attachments/assets/4b8eae0e-919e-441a-8ece-e33b5d5b2e06" />


## Project Workflow
<img width="395" height="225" alt="image" src="https://github.com/user-attachments/assets/ec71b51f-97f6-4034-ac67-49f27312a926" />

## Flowchart Processing Data

<img width="344" height="197" alt="image" src="https://github.com/user-attachments/assets/bf4154fd-6853-4b56-a002-ef29d04453f8" />


## Flowchart Modeling
<img width="323" height="214" alt="image" src="https://github.com/user-attachments/assets/090386a0-4325-444b-8503-01292938ed90" />

## Comparison Table of Model Accuracy and Vectorization
<img width="427" height="185" alt="image" src="https://github.com/user-attachments/assets/6966c601-d78a-47aa-a5b2-af8f7cbcf947" />

Following the identification of the two models with the highest performance scores, the Chi-Square method was subsequently employed.
The Chi-Square method in NLP is used to select features (words/terms) that are most relevant to the category/class, thereby making the model more efficient, accurate, and interpretable.

## Comparison Table of Modeling Results Using Chi-Square
<img width="335" height="78" alt="image" src="https://github.com/user-attachments/assets/4d21257e-03c6-48d8-8c92-976b7b3d25d1" />

Based on the Chi-Square model, the best accuracy was achieved using the SVM model after hyperparameter tuning, with a score of 0.7703, showing an improvement compared to the model without hyperparameter tuning, which scored 0.7686. On the other hand, the Random Forest model with hyperparameter tuning achieved an accuracy of 0.7440, indicating a decrease compared to the model without hyperparameter tuning, which had an accuracy of 0.7507

## Results
- Best Model: SVM with TF-IDF + Chi-Square + Hyperparameter Tuning
- Accuracy: 77.03%
- Business Impact: Faster CV screening with improved candidate-job matching.

## Key Learnings
- Stopword removal did not significantly improve performance.
- Class imbalance caused some bias (overlap between Requirement, Skill, and Soft Skills).
- Future improvement requires real CV training data and relabeling overlapping categories.

## Conclusion
The project successfully delivered an AI-powered recruitment assistant, reducing manual workload and enhancing decision-making with objective, explainable, and scalable ML solutions.

## Flowchart of model usage
<img width="383" height="208" alt="image" src="https://github.com/user-attachments/assets/81f6d5b9-7758-4cfa-ac99-c1613e489114" />

## Model Deployment
<img width="319" height="265" alt="image" src="https://github.com/user-attachments/assets/b82fa07e-24d9-4ef6-8202-35627be605e0" />
The CV Content Classifier application is an implementation of a machine learning model capable of analyzing and categorizing the contents of a CV into relevant categories such as Education, Experience, Skills, and others


<img width="331" height="251" alt="image" src="https://github.com/user-attachments/assets/2664ad53-12b3-4fd0-b91c-37c86e2c0ad0" />

Key Features:
- Input Method: Users can directly input CV text.
- Analysis Process: The model processes the input, identifies important information, and classifies it into predefined categories.
- Output: Presents the analysis results in a way that helps HR or recruiters quickly understand the candidate’s profile.

Data Source : https://www.kaggle.com/datasets/gaveshjain/ford-sentence-classifiaction-dataset 
