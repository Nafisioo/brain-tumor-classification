import json
from pathlib import Path


class History:

    def __init__(self):

        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
        }

    def update(self, train_loss, train_accuracy, val_metrics):

        self.history["train_loss"].append(train_loss)

        self.history["train_accuracy"].append(train_accuracy)

        self.history["val_loss"].append(val_metrics["loss"])

        self.history["val_accuracy"].append(val_metrics["accuracy"])

        self.history["val_precision"].append(val_metrics["precision"])

        self.history["val_recall"].append(val_metrics["recall"])

        self.history["val_f1"].append(val_metrics["f1"])

    def save(self, path):

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:

            json.dump(self.history, f, indent=4)
