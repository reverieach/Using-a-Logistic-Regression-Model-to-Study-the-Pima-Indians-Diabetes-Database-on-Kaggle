import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv('diabetes.csv')

# Replace certain zeros with NaN for cleaning
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df_clean = df.copy()
for col in zero_columns:
    df_clean[col] = df_clean[col].replace(0, np.nan)

# Impute missing values with specified methods
df_clean['Glucose'].fillna(df_clean['Glucose'].mean(), inplace=True)
df_clean['BloodPressure'].fillna(df_clean['BloodPressure'].mean(), inplace=True)
df_clean['SkinThickness'].fillna(df_clean['SkinThickness'].median(), inplace=True)
df_clean['Insulin'].fillna(df_clean['Insulin'].median(), inplace=True)
df_clean['BMI'].fillna(df_clean['BMI'].median(), inplace=True)

# Feature and target selection
X = df_clean.drop('Outcome', axis=1)
y = df_clean['Outcome']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Model training with different train sizes
train_sizes = [0.75, 0.8, 0.85]
results = {}
for train_size in train_sizes:
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, train_size=train_size, random_state=42)
    results[train_size] = (X_train, X_test, y_train, y_test)

for train_size, (X_train, X_test, y_train, y_test) in results.items():
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    print(f"train_size = {train_size}:")
    print(f"  Train accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"  Test accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print()

# Final model (using train_size = 0.8)
X_train, X_test, y_train, y_test = results[0.8]
final_model = LogisticRegression(random_state=42, max_iter=1000)
final_model.fit(X_train, y_train)
y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)

print("Classification report (test set):")
print(classification_report(y_test, y_test_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': final_model.coef_[0],
    'Abs_Coefficient': np.abs(final_model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("\nLogistic Regression Coefficient Ranking:")
print(feature_importance[['Feature', 'Coefficient', 'Abs_Coefficient']])

# Correlation analysis
correlations = df_clean.corr()['Outcome'].sort_values(ascending=False)
print("\nFeature Correlation with Outcome:")
print(correlations)

# Visualization
import matplotlib
matplotlib.use('Agg')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Correlation coefficients bar plot
ax1 = axes[0, 0]
correlations.plot(kind='barh', ax=ax1, color='steelblue')
ax1.set_title('Feature Correlation with Outcome', fontsize=12, fontweight='bold')
ax1.set_xlabel('Correlation Coefficient')
ax1.grid(axis='x', alpha=0.3)

# Logistic regression coefficients bar plot
ax2 = axes[0, 1]
feature_importance_sorted = feature_importance.sort_values('Coefficient')
colors = ['red' if x < 0 else 'green' for x in feature_importance_sorted['Coefficient']]
ax2.barh(feature_importance_sorted['Feature'], feature_importance_sorted['Coefficient'], color=colors)
ax2.set_title('Logistic Regression Coefficients', fontsize=12, fontweight='bold')
ax2.set_xlabel('Coefficient Value')
ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
ax2.grid(axis='x', alpha=0.3)

# Age vs BMI scatter plot
ax3 = axes[1, 0]
scatter_colors = ['red' if val == 1 else 'blue' for val in y]
ax3.scatter(df_clean['Age'], df_clean['BMI'], c=scatter_colors, alpha=0.6, s=30)
ax3.set_xlabel('Age')
ax3.set_ylabel('BMI')
ax3.set_title('Age vs BMI (Blue=No Diabetes, Red=Diabetes)', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)

# Remove the last subplot (no confusion matrix)
fig.delaxes(axes[1, 1])

plt.tight_layout()
plt.savefig('diabetes_analysis.png', dpi=300, bbox_inches='tight')
print("Plots saved: diabetes_analysis.png")
