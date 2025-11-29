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

The analytical workflow begins with loading the Pima Indians Diabetes dataset into a pandas DataFrame. Initial inspection of the data is conducted using descriptive statistics and structural summaries to understand the range and distribution of values in each feature. Exploratory data analysis (EDA) is performed through a comprehensive series of visualizations to identify key patterns such as feature distributions, class imbalance, outliers, and correlations between medical parameters.

Before applying machine learning models, all input features are standardized using `StandardScaler`, as both Logistic Regression and SVM are sensitive to differences in feature scale. Standardization ensures that all features contribute proportionally during model training. Following preprocessing, the dataset is partitioned into training and test sets using an 80:20 split, allowing accurate evaluation of model performance on unseen data.

Two supervised classification models are developed:

- **Logistic Regression:** A linear probability-based classifier that learns a set of weights for each feature by applying Maximum Likelihood Estimation.
- **Support Vector Machine (SVM):** A margin-based classifier that identifies the optimal separating hyperplane to distinguish diabetic versus non-diabetic patients.

Both models are trained on the training subset and subsequently evaluated on the test subset. Standard performance metrics such as accuracy, precision, recall, F1-score, and confusion matrix are calculated. ROC-AUC curves are plotted to assess the ranking and discrimination ability of each model. In addition to numeric metrics, graphical comparisons such as ROC curves, accuracy comparison bars, and confusion matrix heatmaps provide deeper insight into the strengths and limitations of each model.

This workflow ensures a structured pipeline from raw data exploration to predictive modeling and performance assessment, supporting meaningful conclusions about diabetes risk detection.

## Logistic Regression Overview

Logistic Regression is a statistical learning method used for binary classification problems such as predicting whether a patient is diabetic or not. It models the relationship between input features and the probability of class membership. The model forms a linear combination of the input features:

$$z = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n$$

Since the raw output $z$ can take any real value, it is transformed into a probability using the logistic sigmoid function:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

The model parameters $w$ are estimated by minimizing the Binary Cross-Entropy loss, which measures the difference between predicted probabilities and actual labels. The optimization uses gradient descent to iteratively update weights in the direction that reduces loss. A classification threshold (commonly 0.5) converts the predicted probability into class labels.

One major advantage of Logistic Regression is interpretability: the coefficients indicate how strongly each medical feature influences diabetes risk. For instance, in our model, glucose has the highest coefficient, meaning an increase in glucose level significantly increases the probability of diabetes. This transparency makes Logistic Regression well-suited for healthcare applications, where explanatory support is crucial.

## Support Vector Machine Overview

Support Vector Machine (SVM) is a margin-based classification algorithm that aims to find the optimal separating boundary between classes. Instead of focusing on all training samples, SVM concentrates on the most critical points near the decision boundary, called support vectors. The objective is to maximize the margin—the distance between support vectors and the separating hyperplane. A wider margin generally improves generalization to unseen data.

The decision function follows the linear form:

$$f(x) = w^Tx + b$$

Training an SVM involves solving an optimization problem that minimizes the hinge loss:

$$\text{Loss} = \sum \max(0, 1 - y_i(w^Tx_i + b))$$

while simultaneously minimizing the magnitude of the weight vector $||w||$. The balance between correct classification and margin control enhances robustness against noisy or overlapping data. In this project, a linear kernel is used because it is computationally efficient and matches the linear separability observed in our 2D visualization (Glucose vs BMI).

SVM typically performs well on medium-sized medical datasets where avoiding false negatives is important. In predictive diagnosis, a false negative means predicting "healthy" for a diabetic patient—a critical risk. Our results show that SVM produced fewer false negatives than Logistic Regression, justifying its practical advantage.

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

This project explored diabetes prediction using two fundamental machine learning classifiers—Logistic Regression and Support Vector Machine. Through exploratory data analysis, features such as glucose level, BMI, and age were identified as key indicators for diabetes risk. Logistic Regression provided interpretable model coefficients valuable for medical understanding, while SVM produced stronger classification boundaries and reduced false negatives.

Performance evaluation using confusion matrices, ROC-AUC, and precision-recall measures demonstrated that SVM achieved slightly higher accuracy and better recall performance. This suggests that margin-based learning is more effective for this dataset where patient groups overlap in feature space. However, Logistic Regression remains preferable for medical reasoning and model transparency.

Overall, combining clinically interpretable models with strong predictive models can support early diabetes screening and decision-making in healthcare. Future extensions could include hyperparameter tuning, class balancing techniques (SMOTE), and experimenting with non-linear kernels or ensemble models to further improve detection accuracy.

---

**End of Report**
