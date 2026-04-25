import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


df = pd.read_csv("PHISHING.csv")


for col in ['URL', 'FILENAME', 'Title']:
    if col in df.columns:
        df = df.drop(col, axis=1)


df = df.dropna()


print(f"Duplicate rows before removal: {df.duplicated().sum()}")
df = df.drop_duplicates()
print(f"Duplicate rows after removal: {df.duplicated().sum()}")


X = df.drop('label', axis=1)
y = df['label']


categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(exclude=['object']).columns


preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])


pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)


print("\n Model Evaluation Results:")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print(f"\nCross-validation Accuracy: {cv_scores.mean():.4f}")


sample = X_test.iloc[0:1]
actual = y_test.iloc[0]
predicted = pipeline.predict(sample)[0]

print("\n Example Prediction:")
print(f"Actual:    {actual}")
print(f"Predicted: {predicted}")
