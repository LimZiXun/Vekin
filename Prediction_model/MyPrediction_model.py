# previously used in Colab
#pip install scikit-learn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression 

# 1. Load the data
df = pd.read_csv('titanic_dataset.csv')
print(df.columns.tolist())
df.describe()

# 2. Handle missing values for ALL features
feature_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
for col in feature_columns:
    if col in df.columns:
        if df[col].isna().all():  # Entire column is NaN
            if col == 'Sex':
                df[col] = 'male'  # Default value for Sex
            else:
                df[col] = 0  # Default for numeric columns
        elif df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].median())
        else:  # Categorical columns
            df[col] = df[col].fillna(df[col].mode()[0])

# 3. Handle negative 'Fare' values
df['Fare'] = df['Fare'].apply(lambda x: max(0, x))

# 4. Encode categorical variables
if 'Sex' in df.columns:
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

# 5. Create new features
df['FamilySize'] = df['SibSp'] + df['Parch']
df['IsAlone'] = (df['FamilySize'] == 0).astype(int)

# 6. Select relevant features
features = [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
    'FamilySize', 'IsAlone'
]
X = df[features]
y = df['Survived']

# 7. Final NaN check
print("Missing values after preprocessing:")
print(X.isnull().sum())

# 8. Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 9. Normalize with MinMaxScaler
continuous_features = ['Age', 'Fare', 'FamilySize']
scaler = MinMaxScaler()
X_train[continuous_features] = scaler.fit_transform(X_train[continuous_features])
X_test[continuous_features] = scaler.transform(X_test[continuous_features])

# 10. Handle any remaining NaNs
imputer = SimpleImputer(strategy='most_frequent')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# 11. Feature selection
best_features = SelectKBest(score_func=chi2, k=3)
fit = best_features.fit(X_train, y_train)

# 12. Show feature scores
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X_train.columns)
features_scores = pd.concat([df_columns, df_scores], axis=1)
features_scores.columns = ['Features', 'Score']
print(features_scores.sort_values(by='Score', ascending=False))

# 13. Output shapes
print("\nTraining set:", X_train.shape)
print("Testing set:", X_test.shape)

X= df[['Sex', 'Pclass', 'IsAlone']] # the top 3 features
Y= df[['Survived']] # the target output

logreg_model= LogisticRegression()
logreg_model.fit(X_train,y_train)
y_pred=logreg_model.predict(X_test)

from sklearn import metrics
from sklearn.metrics import classification_report
print('Accuracy:',metrics.accuracy_score(y_test, y_pred))
print('Recall:',metrics.recall_score(y_test, y_pred, zero_division=1))
print("Precision:",metrics.precision_score(y_test, y_pred, zero_division=1))
print("CL Report:",metrics.classification_report(y_test, y_pred, zero_division=1))

from sklearn.metrics import f1_score
y_proba = logreg_model.predict_proba(X_test)[:, 1]
best_f1, best_thresh = 0, 0.5
for thresh in np.arange(0.3, 0.71, 0.01):
    y_pred_thresh = (y_proba >= thresh).astype(int)
    score = f1_score(y_test, y_pred_thresh)
    if score > best_f1:
        best_f1, best_thresh = score, thresh
print(f"Best F1: {best_f1}, Best Threshold: {best_thresh}")
