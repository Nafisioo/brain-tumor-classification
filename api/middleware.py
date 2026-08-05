"""
FastAPI exception handlers.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from .exceptions import (
    BrainTumorAPIException,
)


logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI):

    @app.exception_handler(BrainTumorAPIException)
    async def application_exception_handler(
        request: Request,
        exc: BrainTumorAPIException,
    ):

        logger.error(f"{exc.error_code}: {exc.message}")

        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "type": exc.error_code,
                    "message": exc.message,
                    "path": request.url.path,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ):

        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "type": "validation_error",
                    "message": "Invalid request payload.",
                    "details": exc.errors(),
                }
            },
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request,
        exc: Exception,
    ):

        logger.exception(exc)

        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "internal_server_error",
                    "message": "Unexpected server error.",
                }
            },
        )
