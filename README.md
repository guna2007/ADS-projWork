# Diabetes Risk Prediction Using Logistic Regression and SVM

**Authors:** L. Guna, B. Rakesh, IVS. Akhil  
**Roll Numbers:** CS24B2043, CS24B2039, CS24B2005  
**Course:** Applied Data Science  
**Project Title:** Diabetes Risk Prediction Using Logistic Regression and SVM

---

## Introduction

This document presents a comprehensive analysis and predictive modeling study utilizing the Pima Indians Diabetes dataset. The primary objective is to conduct descriptive analytics and develop robust machine learning models for diabetes outcome prediction. The study employs two advanced algorithms—Logistic Regression and Support Vector Machine (SVM)—which extend beyond the scope of standard classroom instruction. The findings elucidate key factors influencing diabetes risk and provide a rigorous evaluation of model performance.

## Objectives

The specific objectives of this project are as follows:

1. To perform descriptive analytics on the Pima Indians Diabetes dataset, thereby elucidating the relationships between medical features and diabetes outcomes.
2. To construct and compare two binary classification models (Logistic Regression and SVM) not previously covered in classroom discussions.
3. To assess the models using established metrics: accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC, and to interpret the results in an academic context.

## Dataset Description

The dataset comprises 768 instances and 9 attributes. The target variable, "Outcome," is binary (0 = non-diabetic, 1 = diabetic). Predictor variables include:  
Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, and Age.  
These features represent medical measurements commonly associated with metabolic health and diabetes risk.

## Folder Structure

```
diabetes_project/
    main.py
    Problem_Statement.txt
    Final_Report.html
    Final_Report.pdf
    README.txt
    requirements.txt
    data/diabetes.csv
    results/(all plot images)
```

## Methodology

The analytical workflow begins with data ingestion and the computation of descriptive statistics. Exploratory data analysis (EDA) is performed through a series of visualizations to examine feature distributions, outliers, correlations, and class balance. All input features are standardized, and the dataset is partitioned into training and test sets using an 80:20 split.

Subsequently, two classification models are developed:

- **Logistic Regression:** A linear probabilistic model employing the sigmoid function and maximum likelihood estimation for binary classification.
- **Support Vector Machine (SVM):** An algorithm that identifies the optimal separating hyperplane by maximizing the margin between classes, implemented with a linear kernel for interpretability.

Model evaluation is conducted using accuracy, confusion matrix, classification report, and ROC-AUC. Visualization techniques are employed to facilitate model comparison and to illustrate feature relationships.

## Logistic Regression Overview

Logistic Regression is a well-established binary classification technique that models the probability of an outcome using the logistic (sigmoid) function. The algorithm estimates coefficients for each predictor, yielding probabilities in the range [0, 1]. A threshold (commonly 0.5) is applied to determine the predicted class. Logistic Regression is valued for its interpretability and efficacy when the relationship between predictors and outcome is approximately linear.

## Support Vector Machine Overview

Support Vector Machine (SVM) is a powerful classification algorithm that seeks the optimal hyperplane to separate classes with maximal margin. By focusing on support vectors—data points nearest to the decision boundary—SVM enhances generalization. In this study, a linear kernel is utilized for clarity and reproducibility. SVM is particularly effective in scenarios with complex or overlapping class boundaries.

## Performance Evaluation

Model performance is assessed using standard metrics: accuracy, precision, recall, F1-score, and ROC-AUC. Confusion matrix heatmaps provide insight into classification errors, while ROC curves visualize ranking performance. Comparative analysis of these metrics enables a rigorous evaluation of both models.

## Results Summary

Upon training and testing, the following results were obtained:

- Logistic Regression: Accuracy = 78.57%
- Support Vector Machine: Accuracy = 79.22%

Additional metrics, including precision, recall, F1-score, and AUC, are reported. Confusion matrices and ROC curves further support the comparative analysis of model efficacy.

## Visualizations and Analytical Insights

### Exploratory Data Analysis

![Glucose Distribution](results/dist_glucose.png)  
_The glucose histogram reveals a right-skewed distribution, indicating a prevalence of high-glucose cases strongly associated with diabetes outcome._

![Missing Values](results/missing_values.png)  
_The missing values chart confirms the absence of null entries. However, certain measurements, such as blood pressure values of zero, likely represent invalid or missing data rather than true zeros._

![BMI Distribution](results/dist_bmi.png)  
_BMI distribution exhibits a slight right skew, with most values falling within the normal to overweight range._

![Age Distribution](results/dist_age.png)  
_Age distribution indicates a majority of younger individuals, with decreasing frequency in older age groups._

![Blood Pressure Distribution](results/dist_bp.png)  
_Blood pressure appears approximately normally distributed, with notable outliers at zero suggesting missing or erroneous measurements._

![Pairplot](results/pairplot.png)  
_Pairplot visualizations highlight relationships among features, with Glucose and BMI demonstrating clear separation between outcome classes._

![Outliers Boxplot](results/outliers_box.png)  
_Boxplots identify outliers in Glucose, BloodPressure, BMI, and Age, with several extreme values present._

![Glucose vs BMI Scatter](results/scatter_glucose_bmi.png)  
_Scatter plots of Glucose versus BMI, colored by outcome, show diabetic cases concentrated in regions of higher glucose and BMI._

![BMI Violin Plot](results/violin_bmi_outcome.png)  
_Violin plots indicate that BMI values are generally higher among diabetic patients compared to non-diabetic individuals._

![Pregnancies Count](results/count_preg.png)  
_Countplots suggest a potential correlation between higher pregnancy counts and increased diabetes risk._

![Class Count](results/class_count.png)  
_Class count plots reveal an imbalance, with the majority class being non-diabetic._

![Correlation Heatmap](results/corr_heatmap.png)  
_Correlation heatmaps highlight Glucose and BMI as strongly associated with diabetes outcome._

### Machine Learning Preparation

![Feature Importance](results/coef_importance.png)  
_Logistic Regression coefficients indicate that Glucose exerts the highest positive influence on diabetes prediction._

![SVM Decision Boundary](results/decision_boundary_svm.png)  
_SVM decision boundary visualizations using Glucose and BMI demonstrate linear separation between classes._

![Correlation Sorted](results/corr_sorted.png)  
_Sorted correlation bar charts confirm Glucose as the feature most strongly correlated with outcome._

![Grouped Mean Features](results/grouped_mean.png)  
_Mean feature values, grouped by class, show that diabetic patients exhibit higher averages for most predictors._

### Model Performance Visualization

![Confusion Matrix Logistic Regression](results/conf_matrix_lr.png)  
_Confusion matrix for Logistic Regression demonstrates a high true negative rate, with a moderate incidence of false negatives._

![Confusion Matrix SVM](results/conf_matrix_svm.png)  
_Confusion matrix for SVM indicates slightly superior performance, with fewer false negatives compared to Logistic Regression._

![ROC Comparison](results/roc_compare.png)  
_ROC curve comparison reveals similar performance for both models, with SVM achieving a marginally higher AUC._

![Accuracy Comparison](results/accuracy_compare.png)  
_Accuracy comparison charts show SVM attaining slightly higher accuracy than Logistic Regression._

![Precision Recall F1 Comparison](results/prec_rec_f1_compare.png)  
_Precision-Recall-F1 score bar graphs indicate comparable precision for both models, with SVM exhibiting improved recall._

## Conclusion

This study demonstrates the application of Logistic Regression and Support Vector Machine for the classification of medical data related to diabetes risk. Logistic Regression is favored for its interpretability, enabling medical professionals to discern the influence of individual predictors. SVM, conversely, is selected for its capacity to delineate complex decision boundaries and deliver enhanced predictive accuracy. The complementary strengths of these models—explainability and performance—underscore their utility in clinical decision support. Future research may explore class balancing techniques and alternative modeling approaches to further improve predictive outcomes.

---

**End of Report**
