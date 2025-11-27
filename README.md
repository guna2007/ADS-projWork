# Diabetes Risk Prediction Using Logistic Regression and SVM

**Name:** L. Guna  
**Roll No:** CS24B2043  
**Course:** Applied Data Science  
**Project Title:** Diabetes Risk Prediction Using Logistic Regression and SVM

---

## Introduction

This report presents an analysis and predictive modeling study performed on the Pima Indians Diabetes dataset. The goal is to explore the dataset through descriptive analytics and build machine learning models that predict diabetes outcome. The study demonstrates the use of two algorithms not covered in classroom instruction: Logistic Regression and Support Vector Machine (SVM). The results help highlight factors influencing diabetes risk and evaluate model effectiveness.

## Objectives

The main objectives of this work are:

1. To perform descriptive analytics on the Pima Indians Diabetes dataset and understand the relationships between medical features and diabetes outcome.
2. To build and compare two binary classification models (Logistic Regression and SVM) that were not covered in classroom discussions.
3. To evaluate the models using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC, and interpret the results.

## Dataset Description

The dataset contains 768 instances and 9 attributes. The target column "Outcome" is binary (0 = non-diabetic, 1 = diabetic). The input features include:  
Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.  
The dataset includes medical measurements often associated with metabolic health and diabetes risk.

## Folder Structure

```
diabetes_project/
    main.py
    Problem_Statement.txt
    Final_Report.html
    Final_Report.pdf
    README.md
    requirements.txt
    data/diabetes.csv
    results/(all plot images)
```

## Code Workflow Explanation

The dataset is loaded and basic descriptive statistics are printed. Various exploratory plots are generated to understand feature distributions, outliers, correlation, and class balance. All input features are standardized using a simple scaling step, and the data is then split into training and test sets using a standard 80:20 split.

Two models are trained:

- **Logistic Regression:** a linear probability classification model based on sigmoid transformation and maximum likelihood.
- **Support Vector Machine:** finds the optimal separating hyperplane maximizing margin between classes.

Both models are evaluated using accuracy, confusion matrix, classification report, and ROC-AUC. Additional visualization plots help compare models and show feature relationships.

## Logistic Regression Explanation

Logistic Regression is a binary classification algorithm that models the probability of an outcome using the logistic (sigmoid) function. The model estimates coefficients for each input feature and outputs a probability between 0 and 1. A threshold (usually 0.5) determines the final predicted class. It is interpretable and effective when the relationship between features and output is approximately linear.

## Support Vector Machine Explanation

SVM is a classification algorithm that identifies the best separating hyperplane to divide classes with the maximum possible margin. By maximizing the distance between support vectors and the separating boundary, SVM improves generalization. In this project, a linear kernel is used for simplicity. SVM focuses on the most influential data points near the boundary.

## Performance Evaluation

Standard evaluation metrics are computed including accuracy, precision, recall, and F1-score. ROC-AUC curves visualize the ranking performance of both models. Confusion matrix heatmaps indicate correct vs incorrect classifications. Comparing model results helps determine which model performs better on this dataset.

## Results Summary

After training the models and evaluating them on the test set, the numerical values of accuracy, precision, recall, F1-score and AUC are reported from the Python output. These values, along with the confusion matrices and ROC curves, are used to compare Logistic Regression and SVM.

In our run, Logistic Regression achieved an accuracy of 78.57% and SVM achieved an accuracy of 79.22%.

## Plots and Inferences

### Exploratory Data Analysis

#### Missing Values

![Missing Values](results/missing_values.png)  
_Missing values chart shows no null entries in the dataset. However, some measurements such as blood pressure having a value of zero are likely to be invalid rather than true zeros, and they behave like hidden missing values._

#### Glucose Distribution

![Glucose Distribution](results/dist_glucose.png)  
_The glucose histogram shows right-skewed distribution indicating many high-glucose cases which strongly relate to diabetes outcome._

#### BMI Distribution

![BMI Distribution](results/dist_bmi.png)  
_BMI distribution shows slight right skew with most values in normal to overweight range._

#### Age Distribution

![Age Distribution](results/dist_age.png)  
_Age distribution shows younger population majority with decreasing frequency in older age groups._

#### Blood Pressure Distribution

![Blood Pressure Distribution](results/dist_bp.png)  
_Blood pressure distribution appears roughly normal with some outliers at zero indicating missing or invalid measurements._

#### Pairplot

![Pairplot](results/pairplot.png)  
_Pairplot shows feature relationships with Glucose and BMI showing clear separation between outcome classes._

#### Outliers Boxplot

![Outliers Boxplot](results/outliers_box.png)  
_Boxplots identify outliers in Glucose, BloodPressure, BMI, and Age with several extreme values present._

#### Glucose vs BMI Scatter

![Glucose vs BMI Scatter](results/scatter_glucose_bmi.png)  
_Scatter plot of Glucose vs BMI colored by outcome shows diabetic cases concentrated in higher glucose and BMI regions._

#### BMI Violin Plot

![BMI Violin Plot](results/violin_bmi_outcome.png)  
_Violin plot shows BMI distribution is higher for diabetic patients compared to non-diabetic._

#### Pregnancies Count

![Pregnancies Count](results/count_preg.png)  
_Countplot shows pregnancy count relationship with outcome indicating higher pregnancy count may correlate with diabetes risk._

#### Class Count

![Class Count](results/class_count.png)  
_Class count plot shows dataset imbalance with majority class being non-diabetic._

#### Correlation Heatmap

![Correlation Heatmap](results/corr_heatmap.png)  
_The correlation heatmap highlights Glucose and BMI as strongly related to diabetes._

### Machine Learning Preparation

#### Feature Importance

![Feature Importance](results/coef_importance.png)  
_Logistic Regression coefficients show Glucose has highest positive influence on diabetes prediction._

#### SVM Decision Boundary

![SVM Decision Boundary](results/decision_boundary_svm.png)  
_SVM decision boundary visualization using Glucose and BMI shows linear separation between classes._

#### Correlation Sorted

![Correlation Sorted](results/corr_sorted.png)  
_Sorted correlation bar chart shows Glucose has strongest positive correlation with outcome._

#### Grouped Mean Features

![Grouped Mean Features](results/grouped_mean.png)  
_Mean feature values grouped by class show diabetic patients have higher average values for most features._

### Model Performance Visualization

#### Confusion Matrix Logistic Regression

![Confusion Matrix Logistic Regression](results/conf_matrix_lr.png)  
_Confusion matrix for Logistic Regression shows good true negative rate but moderate false negative rate._

#### Confusion Matrix SVM

![Confusion Matrix SVM](results/conf_matrix_svm.png)  
_Confusion matrix for SVM shows slightly better performance than Logistic Regression with fewer false negatives._

#### ROC Comparison

![ROC Comparison](results/roc_compare.png)  
_ROC curves comparison shows both models perform similarly with SVM having slightly higher AUC._

#### Accuracy Comparison

![Accuracy Comparison](results/accuracy_compare.png)  
_Accuracy comparison chart shows SVM achieves marginally higher accuracy than Logistic Regression._

#### Precision Recall F1 Comparison

![Precision Recall F1 Comparison](results/prec_rec_f1_compare.png)  
_Precision-Recall-F1 score bar graphs show both models have similar precision but SVM has slightly better recall._

## Conclusion

Both Logistic Regression and SVM perform reasonably well for diabetes classification. Logistic Regression is simpler and more interpretable, whereas SVM provides a maximum-margin boundary that sometimes yields higher recall. Glucose and BMI were observed as major influencing features. This project demonstrates that basic predictive models can assist in early diabetes risk assessment. Future work may explore class balancing techniques or additional models.

---

**End of Report**
