# Customer Complaint Priority Predictor

A machine learning project that predicts the **priority (Low / Medium / High)** of a customer complaint using a Decision Tree.

## Features Used
| Feature | Description |
|---|---|
| `Waiting_Days` | Days the customer has been waiting |
| `Customer_Impact` | Low / Medium / High |
| `Issue_Type` | Query, Billing, Delivery, Payment |

**Target:** `Priority`

## How the Code Works
1. **Data** – A small sample dataset of 20 complaints is created with pandas.
2. **Encoding** – `LabelEncoder` converts text columns into numbers.
3. **Split** – Data is divided into 75% training and 25% testing (stratified).
4. **Model** – A `DecisionTreeClassifier` (max depth = 3) is trained.
5. **Evaluation** – Accuracy and a classification report are printed.
6. **Prediction** – A new complaint (8 days, High impact, Payment) is classified.

## Run
```bash
pip install -r requirements.txt
python Complaint_predictor.py
```

## Sample Output
```
Model Accuracy: ...
New Complaint Prediction: High
```

## Note
The dataset is tiny and synthetic, so results are for learning purposes only.

## Tech Stack
Python, pandas, scikit-learn
