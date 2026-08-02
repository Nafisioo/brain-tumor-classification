"""
Custom application exceptions.

These exceptions are raised inside the ML pipeline
and converted into HTTP responses by middleware.
"""


class BrainTumorAPIException(Exception):
    """
    Base exception for application errors.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "application_error",
    ):
        self.message = message
        self.error_code = error_code

        super().__init__(message)



class ModelNotLoadedError(BrainTumorAPIException):
    """
    Raised when inference model cannot be loaded.
    """

    def __init__(
        self,
        message="Model could not be loaded.",
    ):
        super().__init__(
            message=message,
            error_code="model_not_loaded",
        )



class CheckpointNotFoundError(BrainTumorAPIException):
    """
    Raised when model checkpoint is missing.
    """

    def __init__(
        self,
        message="Model checkpoint not found.",
    ):
        super().__init__(
            message=message,
            error_code="checkpoint_missing",
        )



class InvalidImageError(BrainTumorAPIException):
    """
    Raised when uploaded image cannot be processed.
    """

    def __init__(
        self,
        message="Invalid image file.",
    ):
        super().__init__(
            message=message,
            error_code="invalid_image",
        )



class PredictionError(BrainTumorAPIException):
    """
    Raised when inference fails.
    """

    def __init__(
        self,
        message="Prediction failed.",
    ):
        super().__init__(
            message=message,
            error_code="prediction_failed",
        )