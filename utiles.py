import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

def analyse(y_valid, y_pred):
    TN, FP, FN, TP = confusion_matrix(y_valid, y_pred, labels=[0, 1]).ravel()
    print("TN", TN, "FP", FP, "FN", FN, "TP", TP)
    print("f1:", f1_score(y_valid, y_pred, average='weighted'))

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    encoder = LabelEncoder()
    string_columns = df.select_dtypes(include=['object', 'string']).columns
    for col in string_columns:
        df[col] = encoder.fit_transform(df[col])

    df = df.fillna(-1)
    return df
