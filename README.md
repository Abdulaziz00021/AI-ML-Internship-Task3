# **Heart Disease Prediction – Machine Learning Project**

##  **Task Objective**

Build a machine learning model to predict whether a person is at risk of heart disease based on clinical health data. The goal is to perform EDA, train a classification model, and evaluate its performance using standard metrics.

---

##  **Dataset Used**

**Heart Disease UCI Dataset**
Contains medical attributes such as:

* Age
* Sex
* Chest pain type
* Blood pressure
* Cholesterol
* Max heart rate
* Exercise-induced angina
* ST depression (oldpeak)
* Thal
* Target (1 = heart disease, 0 = no disease)

---

##  **Models Applied**

* **Logistic Regression**
* **Decision Tree Classifier** (optional alternative)

---

##  **Key Results & Findings**

* Logistic Regression achieved **strong accuracy** on the test set.
* **Confusion matrix** showed balanced classification.
* **ROC curve** indicated a high AUC score.
* Important features contributing to prediction:

  * Maximum heart rate (thalach)
  * ST depression (oldpeak)
  * Chest pain type (cp)
  * Sex
  * Exercise-induced angina (exang)

