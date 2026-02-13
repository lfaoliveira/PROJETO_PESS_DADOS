import pandas as pd

import matplotlib.pyplot as plt


def plot_classification_errors(df: pd.DataFrame, target_col: str, pred_col: str):
    """
    Categorizes results into True Positive, True Negative, False Positive, and False Negative,
    then generates a bar chart for analysis.
    """
    conditions = [
        (df[target_col] == 1) & (df[pred_col] == 1),
        (df[target_col] == 0) & (df[pred_col] == 0),
        (df[target_col] == 0) & (df[pred_col] == 1),
        (df[target_col] == 1) & (df[pred_col] == 0),
    ]
    choices = [
        "True Positive",
        "True Negative",
        "False Positive",
        "False Negative",
    ]

    df["classification_type"] = (
        pd.Series(pd.NA, index=df.index)
        .mask(conditions[0], choices[0])
        .mask(conditions[1], choices[1])
        .mask(conditions[2], choices[2])
        .mask(conditions[3], choices[3])
    )

    error_counts = (
        df["classification_type"].value_counts().reindex(choices, fill_value=0)
    )

    plt.figure(figsize=(10, 6))
    colors = ["#2ca02c", "#1f77b4", "#d62728", "#ff7f0e"]
    error_counts.plot(kind="bar", color=colors)

    plt.title("Validation Error Analysis: Classification Outcomes")
    plt.ylabel("Count")
    plt.xlabel("Category")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
