from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def calculate_metrics(predictions, labels):

    metrics = {}

    metrics["accuracy"] = accuracy_score(labels, predictions)

    metrics["precision"] = precision_score(
        labels, predictions, average="macro", zero_division=0
    )

    metrics["recall"] = recall_score(
        labels, predictions, average="macro", zero_division=0
    )

    metrics["f1"] = f1_score(labels, predictions, average="macro", zero_division=0)

    metrics["confusion_matrix"] = confusion_matrix(labels, predictions)

    return metrics
