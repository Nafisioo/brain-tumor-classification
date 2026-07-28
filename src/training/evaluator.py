import torch

from src.utils.metrics import calculate_metrics



@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device
):
    """
    Evaluate model.

    Returns:
        metrics:
            loss
            accuracy
            precision
            recall
            f1
            confusion_matrix

            predictions
            labels
            paths
            confidence
    """


    model.eval()


    total_loss = 0.0


    y_pred = []
    y_true = []

    confidences = []

    all_paths = []



    for batch in loader:
        if len(batch) == 3:
            images, labels, paths = batch
        else:
            images, labels = batch
            paths = [""] * len(images)

        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        loss = criterion(outputs, labels)
        total_loss += loss.item()

        probabilities = torch.softmax(outputs, dim=1)
        confidence, predictions = torch.max(probabilities, dim=1)

        y_pred.extend(predictions.cpu().numpy())
        y_true.extend(labels.cpu().numpy())
        confidences.extend(confidence.cpu().numpy())
        all_paths.extend(paths)
    metrics = calculate_metrics(
        y_pred,
        y_true
    )



    metrics.update({
        "loss": total_loss / len(loader),
        "predictions": y_pred,
        "labels": y_true,
        "paths": all_paths,
        "confidence": confidences
    })
    return metrics