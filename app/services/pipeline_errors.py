class PipelineStageError(RuntimeError):
    def __init__(
        self,
        stage: str,
        message: str,
        *,
        code: str = "pipeline_stage_failed",
        status_code: int = 500,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.message = message
        self.code = code
        self.status_code = status_code

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "stage": self.stage,
            "message": self.message,
        }
