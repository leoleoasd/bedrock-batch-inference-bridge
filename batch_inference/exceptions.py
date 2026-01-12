"""Custom exceptions for batch inference SDK."""


class BatchInferenceError(Exception):
    """Base exception for batch inference errors."""

    pass


class BatchJobError(BatchInferenceError):
    """Raised when a batch job fails."""

    def __init__(self, job_arn: str, status: str, message: str = ""):
        self.job_arn = job_arn
        self.status = status
        super().__init__(f"Batch job {job_arn} failed with status {status}: {message}")


class InvocationError(BatchInferenceError):
    """Raised when an individual invocation within a batch fails."""

    def __init__(self, record_id: str, error_code: str, error_message: str):
        self.record_id = record_id
        self.error_code = error_code
        self.error_message = error_message
        super().__init__(f"Invocation {record_id} failed: {error_code} - {error_message}")


class NotSetupError(BatchInferenceError):
    """Raised when invoke_model is called before setup."""

    def __init__(self):
        super().__init__("Client not set up. Call setup() before invoke_model().")


class AlreadySetupError(BatchInferenceError):
    """Raised when setup is called more than once."""

    def __init__(self):
        super().__init__("Client already set up. Create a new client instance.")
