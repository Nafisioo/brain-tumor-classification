from pathlib import Path


class ResNetConfig:

    # data

    image_size = 224

    num_classes = 4


    # training

    epochs = 20

    batch_size = 32


    # stage 1

    classifier_lr = 1e-3


    # stage 2

    backbone_lr = 1e-5

    fine_tune_lr = 1e-4



    weight_decay = 1e-4



    dropout = 0.3



    seed = 42



    # experiment paths


    experiment_name = (
        "resnet18_finetune"
    )


    checkpoint_dir = Path(
        "experiments/resnet18_finetune/checkpoints"
    )


    log_file = Path(
        "experiments/resnet18_finetune/logs/training_history.json"
    )


    result_dir = Path(
        "experiments/resnet18_finetune/results"
    )


config = ResNetConfig()