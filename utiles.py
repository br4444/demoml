import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (confusion_matrix, 
                             classification_report,
                             f1_score,
                             roc_curve,
                             auc)

def analyse(y_valid, y_pred):
    TN, FP, FN, TP = confusion_matrix(y_valid, y_pred, labels=[0, 1]).ravel()
    print("TN", TN, "FP", FP, "FN", FN, "TP", TP)
    print(classification_report(y_valid, y_pred))
    print("f1:", round(f1_score(y_valid, y_pred, average='weighted'),3))
    # fpr, tpr, _ = roc_curve(y_valid, y_pred)
    # print(f"auc: {round(auc(fpr, tpr), 2)}")

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    encoder = LabelEncoder()
    string_columns = df.select_dtypes(include=['object', 'string']).columns
    for col in string_columns:
        df[col] = encoder.fit_transform(df[col])

    df = df.fillna(-1)
    return df
