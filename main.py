import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

if not os.path.exists("results"):
    os.makedirs("results")

df = pd.read_csv("data/diabetes.csv")

print(df.head())
print(df.describe())
print(df["Outcome"].value_counts())

plt.figure()
df["Glucose"].hist()
plt.title("Glucose Histogram")
plt.savefig("results/glucose_hist.png")

plt.figure()
sns.boxplot(y=df["BMI"])
plt.title("BMI Boxplot")
plt.savefig("results/bmi_boxplot.png")

plt.figure()
df["Outcome"].value_counts().plot(kind="bar")
plt.title("Class Count")
plt.savefig("results/class_count.png")

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.savefig("results/corr_heatmap.png")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=7)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

print("Logistic Regression")
print("Accuracy:", accuracy_score(y_test, log_pred))
print(confusion_matrix(y_test, log_pred))
print(classification_report(y_test, log_pred))

log_prob = log_model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, log_prob)
log_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr)
plt.title("ROC Logistic Regression")
plt.savefig("results/logistic_roc.png")

svm_model = SVC(kernel="linear", probability=True)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

print("SVM")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print(confusion_matrix(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

svm_prob = svm_model.predict_proba(X_test)[:,1]
fpr_s, tpr_s, _ = roc_curve(y_test, svm_prob)
svm_auc = auc(fpr_s, tpr_s)

plt.figure()
plt.plot(fpr_s, tpr_s)
plt.title("ROC SVM")
plt.savefig("results/svm_roc.png")

# EDA plots
# missing values
plt.figure()
df.isnull().sum().plot(kind="bar")
plt.title("Missing Values")
plt.savefig("results/missing_values.png")

# dist plots
plt.figure()
df["Glucose"].hist(bins=20, edgecolor="black")
plt.title("Glucose Distribution")
plt.savefig("results/dist_glucose.png")

plt.figure()
df["BMI"].hist(bins=20, edgecolor="black")
plt.title("BMI Distribution")
plt.savefig("results/dist_bmi.png")

plt.figure()
df["Age"].hist(bins=20, edgecolor="black")
plt.title("Age Distribution")
plt.savefig("results/dist_age.png")

plt.figure()
df["BloodPressure"].hist(bins=20, edgecolor="black")
plt.title("BloodPressure Distribution")
plt.savefig("results/dist_bp.png")

# pairplot
pairplot_fig = sns.pairplot(df[["Glucose", "BMI", "Age", "Outcome"]], hue="Outcome")
pairplot_fig.savefig("results/pairplot.png")

# outliers boxplot
fig, axes = plt.subplots(2, 2, figsize=(10,8))
sns.boxplot(y=df["Glucose"], ax=axes[0,0])
axes[0,0].set_title("Glucose Outliers")
sns.boxplot(y=df["BloodPressure"], ax=axes[0,1])
axes[0,1].set_title("BloodPressure Outliers")
sns.boxplot(y=df["BMI"], ax=axes[1,0])
axes[1,0].set_title("BMI Outliers")
sns.boxplot(y=df["Age"], ax=axes[1,1])
axes[1,1].set_title("Age Outliers")
plt.tight_layout()
plt.savefig("results/outliers_box.png")

# scatter glucose vs bmi
plt.figure()
plt.scatter(df["Glucose"], df["BMI"], c=df["Outcome"], cmap="coolwarm", alpha=0.6)
plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.title("Glucose vs BMI")
plt.colorbar()
plt.savefig("results/scatter_glucose_bmi.png")

# violin bmi outcome
plt.figure()
sns.violinplot(x=df["Outcome"], y=df["BMI"])
plt.title("BMI by Outcome")
plt.savefig("results/violin_bmi_outcome.png")

# pregnancies count
plt.figure()
sns.countplot(x="Pregnancies", hue="Outcome", data=df)
plt.title("Pregnancies vs Outcome")
plt.savefig("results/count_preg.png")

# ML prep plots
# feature importance from logistic regression coef
plt.figure()
coef = log_model.coef_[0]
features = X.columns
plt.barh(features, coef)
plt.title("Logistic Regression Coefficients")
plt.savefig("results/coef_importance.png")

# decision boundary svm for glucose and bmi
df_temp = df[["Glucose", "BMI", "Outcome"]].copy()
X_temp = df_temp[["Glucose", "BMI"]].values
y_temp = df_temp["Outcome"].values
svm_temp = SVC(kernel="linear")
svm_temp.fit(X_temp, y_temp)

h = 1
x_min, x_max = X_temp[:,0].min()-10, X_temp[:,0].max()+10
y_min, y_max = X_temp[:,1].min()-5, X_temp[:,1].max()+5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = svm_temp.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
plt.scatter(X_temp[:,0], X_temp[:,1], c=y_temp, cmap="coolwarm", edgecolor="k")
plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.title("SVM Decision Boundary")
plt.savefig("results/decision_boundary_svm.png")

# correlation sorted
plt.figure()
corr_outcome = df.corr()["Outcome"].drop("Outcome").sort_values()
corr_outcome.plot(kind="barh")
plt.title("Correlation with Outcome")
plt.savefig("results/corr_sorted.png")

# grouped mean
plt.figure()
top_features = ["Glucose", "BMI", "Age", "Insulin"]
df_grouped = df.groupby("Outcome")[top_features].mean()
df_grouped.T.plot(kind="bar")
plt.title("Mean Feature Values by Class")
plt.savefig("results/grouped_mean.png")

# model performance plots
# confusion matrices
cm_lr = confusion_matrix(y_test, log_pred)
plt.figure()
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix LR")
plt.savefig("results/conf_matrix_lr.png")

cm_svm = confusion_matrix(y_test, svm_pred)
plt.figure()
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix SVM")
plt.savefig("results/conf_matrix_svm.png")

# combined roc
plt.figure()
plt.plot(fpr, tpr, label=f"LR AUC={log_auc:.2f}")
plt.plot(fpr_s, tpr_s, label=f"SVM AUC={svm_auc:.2f}")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Comparison")
plt.legend()
plt.savefig("results/roc_compare.png")

# accuracy compare
plt.figure()
acc_lr = accuracy_score(y_test, log_pred)
acc_svm = accuracy_score(y_test, svm_pred)
plt.bar(["Logistic Regression", "SVM"], [acc_lr, acc_svm])
plt.ylim(0, 1)
plt.title("Accuracy Comparison")
plt.savefig("results/accuracy_compare.png")

# precision recall f1
from sklearn.metrics import precision_score, recall_score, f1_score
prec_lr = precision_score(y_test, log_pred)
rec_lr = recall_score(y_test, log_pred)
f1_lr = f1_score(y_test, log_pred)

prec_svm = precision_score(y_test, svm_pred)
rec_svm = recall_score(y_test, svm_pred)
f1_svm = f1_score(y_test, svm_pred)

metrics = ["Precision", "Recall", "F1"]
lr_vals = [prec_lr, rec_lr, f1_lr]
svm_vals = [prec_svm, rec_svm, f1_svm]

x = np.arange(len(metrics))
width = 0.35

plt.figure()
plt.bar(x - width/2, lr_vals, width, label="LR")
plt.bar(x + width/2, svm_vals, width, label="SVM")
plt.ylabel("Score")
plt.title("Precision Recall F1 Comparison")
plt.xticks(x, metrics)
plt.legend()
plt.savefig("results/prec_rec_f1_compare.png")

print("Done")
