from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np



def create_train_val_indices(
    dataset,
    validation_split,
    seed
):

    targets = np.array(
        dataset.targets
    )


    indices = np.arange(
        len(dataset)
    )


    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_split,
        random_state=seed
    )


    train_indices, val_indices = next(
        splitter.split(
            indices,
            targets
        )
    )


    return (
        train_indices.tolist(),
        val_indices.tolist()
    )