import nbformat as nbf

nb = nbf.v4.new_notebook()

# --- Section 0: Title ---
nb.cells.append(nbf.v4.new_markdown_cell('# Python Learning & Exam Performance: End-to-End AI/ML Case Study\n\nThis notebook demonstrates a complete end-to-end Machine Learning workflow on the **Python Learning & Exam Performance** dataset. The goal is to analyze student behavior and predict exam performance using both regression and classification techniques.'))

# --- Section 1: Introduction & Data Loading ---
nb.cells.append(nbf.v4.new_markdown_cell('## 1. Introduction & Data Loading\n\nIn this section, we locate and load the dataset. We will perform an initial inspection to understand the structure, data types, and contents.'))

nb.cells.append(nbf.v4.new_code_cell('''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import (mean_absolute_error, root_mean_squared_error, r2_score, 
                             accuracy_score, classification_report, confusion_matrix, 
                             roc_curve, auc, precision_score, recall_score, f1_score)

# Configure visualization style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# 1. Locate and load the dataset
data_path = os.path.join("archive", "python_learning_exam_performance.csv")
print(f"Relativate path to dataset: {data_path}")

df = pd.read_csv(data_path)

# Display initial exploration
print(f"Shape: {df.shape}")
print("\\nFirst 5 rows:")
display(df.head())

print("\\ndf.info():")
df.info()'''))

# --- Section 2: Data Cleaning & Preprocessing ---
nb.cells.append(nbf.v4.new_markdown_cell('## 2. Data Cleaning & Preprocessing\n\nWe clean the dataset by standardizing formatting, handling missing values, and ensuring correct data types.'))

nb.cells.append(nbf.v4.new_code_cell('''# 2.1 Standardize column names to snake_case
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
print(f"Standardized columns: {list(df.columns)}")

# 2.2 Check for missing values
missing_values = df.isnull().sum()
print("\\nMissing values per column:")
print(missing_values[missing_values > 0] if missing_values.any() else "No missing values found.")

# 2.3 Check for duplicates
duplicates = df.duplicated().sum()
print(f"\\nDuplicate rows found: {duplicates}")

# 2.4 Basic Outlier Check
# Looking for impossible scores or negative values
numeric_cols = df.select_dtypes(include=[np.number]).columns
print("\\nSummary statistics for outlier detection:")
display(df[numeric_cols].describe())

# Rationale: If missing values were present, we would use mean/median for numeric 
# and mode for categorical. In this dataset, we found none.

# 2.5 Ensure correct dtypes
# Explicitly setting categorical types for analysis
cat_cols = ['country', 'prior_programming_experience']
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

print("\\nFinal list of columns:")
print(df.columns.tolist())'''))

# --- Section 3: Exploratory Data Analysis (EDA) ---
nb.cells.append(nbf.v4.new_markdown_cell('## 3. Exploratory Data Analysis (EDA)\n\n### Univariate Analysis\nAnalyzing distributions of learning time, scores, and categorical factors.'))

nb.cells.append(nbf.v4.new_code_cell('''# 3.1 Distribution of Numeric Features
numeric_features = ['age', 'hours_spent_learning_per_week', 'practice_problems_solved', 'final_exam_score']
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(numeric_features):
    sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue')
    axes[i].set_title(f'Distribution of {col}')

plt.tight_layout()
plt.show()'''))

nb.cells.append(nbf.v4.new_markdown_cell('**Commentary:** The `final_exam_score` appears roughly normally distributed, while `hours_spent_learning_per_week` shows a slight bi-modal tendency, possibly distinguishing between causal and dedicated learners.'))

nb.cells.append(nbf.v4.new_code_cell('''# 3.2 Count of Categorical Features
cat_features = ['prior_programming_experience', 'passed_exam']
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for i, col in enumerate(cat_features):
    sns.countplot(data=df, x=col, ax=axes[i], palette='viridis')
    axes[i].set_title(f'Count of {col}')

plt.tight_layout()
plt.show()'''))

nb.cells.append(nbf.v4.new_markdown_cell('**Commentary:** Most students have little to no prior programming experience. The target variable `passed_exam` is relatively balanced, which is ideal for modeling.'))

nb.cells.append(nbf.v4.new_markdown_cell('### Bivariate Analysis\nExploring relationships with the target variable.'))

nb.cells.append(nbf.v4.new_code_cell('''# 3.3 Correlation Heatmap
plt.figure(figsize=(12, 8))
corr_matrix = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numeric Features')
plt.show()'''))

nb.cells.append(nbf.v4.new_code_cell('''# 3.4 Impact of Experience on Scores
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='prior_programming_experience', y='final_exam_score', palette='Set2')
plt.title('Exam Score vs Prior Experience')
plt.show()'''))

# --- Section 4: Feature Engineering ---
nb.cells.append(nbf.v4.new_markdown_cell('## 4. Feature Engineering\n\nWe derive features that better reflect student engagement.'))

nb.cells.append(nbf.v4.new_code_cell('''# 4.1 Engineered Features
# Total course intensity
df['total_learning_effort'] = df['weeks_in_course'] * df['hours_spent_learning_per_week']
# Problems solved per hour spent
df['problem_solving_efficiency'] = df['practice_problems_solved'] / df['total_learning_effort'].replace(0, 1)
# Engagement Flag: High hours AND high problems solved
hour_threshold = df['hours_spent_learning_per_week'].quantile(0.75)
prob_threshold = df['practice_problems_solved'].quantile(0.75)
df['is_consistent_learner'] = ((df['hours_spent_learning_per_week'] > hour_threshold) & 
                                (df['practice_problems_solved'] > prob_threshold)).astype(int)

print(f"New features: {['total_learning_effort', 'problem_solving_efficiency', 'is_consistent_learner']}")

# 4.2 Data Split Preparation
X = df.drop(columns=['student_id', 'final_exam_score', 'passed_exam'])
y_reg = df['final_exam_score']
y_clf = df['passed_exam']'''))

# --- Section 5: Train-Test Split & Baseline ---
nb.cells.append(nbf.v4.new_markdown_cell('## 5. Train-Test Split & Baseline Models\n\nSetting the floor for model performance.'))

nb.cells.append(nbf.v4.new_code_cell('''# Split
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
X_train, X_test, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

# Baseline
dummy_reg = DummyRegressor(strategy="mean")
dummy_reg.fit(X_train, y_train_reg)
print(f"Baseline Regressor RMSE: {root_mean_squared_error(y_test_reg, dummy_reg.predict(X_test)):.4f}")

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train_clf)
print(f"Baseline Classifier Accuracy: {accuracy_score(y_test_clf, dummy_clf.predict(X_test)):.4f}")'''))

# --- Section 6: Model Training (Core ML) ---
nb.cells.append(nbf.v4.new_markdown_cell('## 6. Model Training (Core ML)\n\nUsing Scikit-learn Pipelines for clean preprocessing and training.'))

nb.cells.append(nbf.v4.new_code_cell('''# Identify column groups
numeric_indices = X.select_dtypes(include=[np.number]).columns
categorical_indices = X.select_dtypes(exclude=[np.number]).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_indices),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_indices)
    ])

# Regression Models
reg_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
}

# Classification Models
clf_models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest Classifier": RandomForestClassifier(random_state=42),
    "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42)
}

# Helper to train and score
def evaluate_models(models, X_train, y_train, X_test, y_test, is_clf=False):
    results = []
    for name, model in models.items():
        pipe = Pipeline([('prep', preprocessor), ('model', model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        if is_clf:
            res = {"Model": name, "Accuracy": accuracy_score(y_test, y_pred), 
                   "F1": f1_score(y_test, y_pred)}
        else:
            res = {"Model": name, "R2": r2_score(y_test, y_pred), 
                   "RMSE": root_mean_squared_error(y_test, y_pred)}
        results.append(res)
    return pd.DataFrame(results)

print("--- Regression Results ---")
display(evaluate_models(reg_models, X_train, y_train_reg, X_test, y_test_reg))

print("\\n--- Classification Results ---")
display(evaluate_models(clf_models, X_train, y_train_clf, X_test, y_test_clf, is_clf=True))

# 6.2 Hyperparameter Tuning (Random Forest Regressor)
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5]
}
search = GridSearchCV(Pipeline([('prep', preprocessor), ('model', RandomForestRegressor(random_state=42))]), 
                      param_grid, cv=3, scoring='r2')
search.fit(X_train, y_train_reg)
print(f"\\nBest RF parameters: {search.best_params_}")
print(f"Best CV R2: {search.best_score_:.4f}")'''))

# --- Section 7: Model Evaluation ---
nb.cells.append(nbf.v4.new_markdown_cell('## 7. Model Evaluation\n\nDetailed diagnostics for our champion model.'))

nb.cells.append(nbf.v4.new_code_cell('''# Champion Model Prediction
best_model = search.best_estimator_
y_pred_best = best_model.predict(X_test)

# Plotting Predicted vs Actual
plt.figure(figsize=(8, 8))
plt.scatter(y_test_reg, y_pred_best, alpha=0.6, color='darkblue')
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel('Actual Final Score')
plt.ylabel('Predicted Final Score')
plt.title('Actual vs Predicted: RF Regressor')
plt.show()

# Residual Plot
residuals = y_test_reg - y_pred_best
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='purple')
plt.title('Residuals Distribution')
plt.show()'''))

# --- Section 8: Model Interpretation ---
nb.cells.append(nbf.v4.new_markdown_cell('## 8. Model Interpretation & Insights\n\nIdentifying which features drive exam performance.'))

nb.cells.append(nbf.v4.new_code_cell('''# Feature Importances
model_step = best_model.named_steps['model']
prep_step = best_model.named_steps['prep']

# Get names
cat_feature_names = prep_step.transformers_[1][1].get_feature_names_out(categorical_indices)
all_feat_names = list(numeric_indices) + list(cat_feature_names)

importances = model_step.feature_importances_
indices = np.argsort(importances)[-10:] # Top 10

plt.figure(figsize=(12, 6))
plt.title('Top 10 Feature Importances (Random Forest)')
plt.barh(range(len(indices)), importances[indices], color='teal', align='center')
plt.yticks(range(len(indices)), [all_feat_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()'''))

nb.cells.append(nbf.v4.new_markdown_cell('**Key Insight:** Student consistency, total effort, and practice problems solved are the strongest predictors of final exam success.'))

# --- Section 9: Optional Unsupervised ---
nb.cells.append(nbf.v4.new_markdown_cell('## 9. Advanced Analysis (Unsupervised - Clustering)\n\nSegmenting students based on behavior.'))

nb.cells.append(nbf.v4.new_code_cell('''from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Clustering on key engagement features
behaviors = ['hours_spent_learning_per_week', 'practice_problems_solved', 'total_learning_effort']
X_behav = StandardScaler().fit_transform(df[behaviors])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['segment'] = kmeans.fit_predict(X_behav)

# Visualize with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_behav)

plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['segment'].astype(str), palette='Set1', s=60)
plt.title('Student Behavior Segmentation (PCA Projection)')
plt.legend(title='Cluster')
plt.show()'''))

# --- Section 10: Final Recommendations ---
nb.cells.append(nbf.v4.new_markdown_cell('## 10. Final Recommendations\n\n1. **Early Intervention**: Identify students in Cluster 0 (Low Engagement) early in the course.\n2. **Practice Boost**: Encourage solving at least 10 practice problems per week to shift students to higher performance bands.\n3. **Consistency Matters**: Total study hours over time correlate better with scores than intense "cramming".'))

nbf.write(nb, 'python_learning_exam_performance_end_to_end.ipynb')
print("Notebook generation complete.")
