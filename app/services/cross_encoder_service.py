import asyncio
import os
from threading import RLock


class CrossEncoderService:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._lock = RLock()

    def _load_model(self):
        with self._lock:
            if self._model is None:
                os.environ.setdefault("USE_TF", "0")
                os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
                try:
                    from sentence_transformers import CrossEncoder
                except ImportError as exc:
                    raise RuntimeError(
                        "sentence-transformers is not installed. "
                        "Use Python 3.11 and run `pip install -r requirements.txt`."
                    ) from exc

                self._model = CrossEncoder(self.model_name)

        return self._model

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        with self._lock:
            scores = self._load_model().predict(pairs)
        return [float(score) for score in scores]

    async def score_pairs_async(self, pairs: list[tuple[str, str]]) -> list[float]:
        return await asyncio.to_thread(self.score_pairs, pairs)
