from inference.predictor import BrainTumorPredictor


def test_model_loading():

    predictor = BrainTumorPredictor()

    assert predictor.model is not None

    assert predictor.model_name == ("resnet18_finetune")

    assert len(predictor.class_names) == 4


def test_model_is_inference_mode():

    predictor = BrainTumorPredictor()

    assert predictor.model.training is False
