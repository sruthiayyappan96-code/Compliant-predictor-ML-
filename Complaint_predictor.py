
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Sample customer complaint data
data = {
    "Waiting_Days": [1, 2, 5, 7, 1, 8, 3, 6, 2, 9,
                     4, 10, 1, 5, 7, 3, 8, 2, 6, 4],
    "Customer_Impact": [
        "Low", "Low", "Medium", "High", "Medium",
        "High", "Low", "High", "Medium", "High",
        "Medium", "High", "Low", "Medium", "High",
        "Low", "High", "Medium", "High", "Low"
    ],
    "Issue_Type": [
        "Query", "Billing", "Delivery", "Payment", "Billing",
        "Payment", "Query", "Delivery", "Billing", "Payment",
        "Delivery", "Payment", "Query", "Billing", "Delivery",
        "Query", "Payment", "Billing", "Delivery", "Query"
    ],
    "Priority": [
        "Low", "Low", "Medium", "High", "Medium",
        "High", "Low", "High", "Medium", "High",
        "Medium", "High", "Low", "Medium", "High",
        "Low", "High", "Medium", "High", "Low"
    ]
}

df = pd.DataFrame(data)

# 2. Convert text into numbers
encoders = {}

for column in ["Customer_Impact", "Issue_Type", "Priority"]:
    encoders[column] = LabelEncoder()
    df[column] = encoders[column].fit_transform(df[column])

# 3. Select input features (X) and target (y)
X = df[["Waiting_Days", "Customer_Impact", "Issue_Type"]]
y = df["Priority"]

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 5. Train the ML model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate the model
predictions = model.predict(X_test)

print("Model Accuracy:", round(accuracy_score(y_test, predictions), 2))
print("\nEvaluation Report:")
print(classification_report(
    y_test, predictions,
    labels=encoders["Priority"].transform(["High", "Low", "Medium"]),
    target_names=["High", "Low", "Medium"],
    zero_division=0
))

# 7. Predict a new complaint
new_complaint = pd.DataFrame({
    "Waiting_Days": [8],
    "Customer_Impact": [
        encoders["Customer_Impact"].transform(["High"])[0]
    ],
    "Issue_Type": [
        encoders["Issue_Type"].transform(["Payment"])[0]
    ]
})

result = model.predict(new_complaint)[0]
priority = encoders["Priority"].inverse_transform([result])[0]

print("\nNew Complaint Prediction:", priority)
