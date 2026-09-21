import pandas as pd

# Load dataset
df = pd.read_csv("dataset/winequality-red.csv", sep=";")

# Display first 5 rows
print(df.head())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

print("\nShape After Removing Duplicates:")
print(df.shape)
print("\nQuality Value Counts:")
print(df["quality"].value_counts().sort_index())

# Create classification target
df["quality_label"] = df["quality"].apply(
    lambda x: 1 if x > 5 else 0
)

print("\nQuality Label Counts:")
print(df["quality_label"].value_counts())

print("\nSample Quality and Label:")
print(df[["quality", "quality_label"]].head(10))

# Separate features and target

X = df.drop(columns=["quality", "quality_label"])
y = df["quality_label"]

print("\nFeatures Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

print("\nFeature Columns:")
print(X.columns)

from sklearn.model_selection import train_test_split

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:")
print(X_train.shape)

print("\nTesting Data:")
print(X_test.shape)

print("\nTraining Target:")
print(y_train.shape)

print("\nTesting Target:")
print(y_test.shape)

from sklearn.preprocessing import StandardScaler

# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaled Training Data Shape:")
print(X_train_scaled.shape)

print("\nScaled Testing Data Shape:")
print(X_test_scaled.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nLogistic Regression Accuracy:")
print(accuracy)
print(f"Accuracy Percentage: {accuracy * 100:.2f}%")

from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Bad", "Good"]
))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

from sklearn.tree import DecisionTreeClassifier

# Create Decision Tree model
decision_tree = DecisionTreeClassifier(
    random_state=42
)

# Train the model
decision_tree.fit(X_train_scaled, y_train)

# Make predictions
y_pred_tree = decision_tree.predict(X_test_scaled)

# Calculate accuracy
tree_accuracy = accuracy_score(y_test, y_pred_tree)

print("\nDecision Tree Accuracy:")
print(tree_accuracy)
print(f"Accuracy Percentage: {tree_accuracy * 100:.2f}%")

from sklearn.ensemble import RandomForestClassifier

# Create Random Forest model
random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
random_forest.fit(X_train_scaled, y_train)

# Make predictions
y_pred_rf = random_forest.predict(X_test_scaled)

# Calculate accuracy
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print("\nRandom Forest Accuracy:")
print(rf_accuracy)
print(f"Accuracy Percentage: {rf_accuracy * 100:.2f}%")

from sklearn.metrics import classification_report, confusion_matrix

print("\nRandom Forest Classification Report:")
print(classification_report(
    y_test,
    y_pred_rf,
    target_names=["Bad", "Good"]
))

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

import joblib

# Save the trained Random Forest model
joblib.dump(random_forest, "model/random_forest_model.pkl")

# Save the scaler
joblib.dump(scaler, "model/scaler.pkl")

print("\nModel and Scaler saved successfully!")