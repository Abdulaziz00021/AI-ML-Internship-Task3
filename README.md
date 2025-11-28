 **Task Objective**

Build a machine learning model to predict whether a person is at risk of heart disease based on clinical health data. The goal is to perform EDA, train a classification model, and evaluate its performance using standard metrics.

 **Dataset Used**

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



 **Models Applied**

* **Logistic Regression** (primary model)
* Optional alternative model: **Decision Tree Classifier**

---

 **Key Results & Findings**

* The Logistic Regression model achieved **good accuracy** on the test dataset.
* **Confusion matrix** showed balanced performance across both classes.
* **ROC curve** indicated strong model separation with a high AUC score.
* Most important features influencing prediction included:

  * Maximum heart rate (thalach)
  * ST depression (oldpeak)
  * Chest pain type (cp)
  * Sex
  * Exercise-induced angina (exang)


