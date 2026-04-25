import asyncio
import os
from threading import RLock

from app.core.config import get_settings


class EmbeddingService:
    """Thin wrapper around SentenceTransformers for local embedding generation."""

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model_name
        self._model = None
        self._model_lock = RLock()

    def _load_model(self):
        with self._model_lock:
            if self._model is None:
                os.environ.setdefault("USE_TF", "0")
                os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise RuntimeError(
                        "sentence-transformers is not installed. "
                        "Use Python 3.11 and run `pip install -r requirements.txt`."
                    ) from exc

                self._model = SentenceTransformer(self.model_name)

        return self._model

    def embed_texts(self, texts: list[str]):
        if not texts:
            raise ValueError("`texts` must contain at least one item.")

        with self._model_lock:
            vectors = self._load_model().encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return vectors.astype("float32")

    def embed_text(self, text: str):
        return self.embed_texts([text])[0]

    async def embed_texts_async(self, texts: list[str]):
        return await asyncio.to_thread(self.embed_texts, texts)

    async def embed_text_async(self, text: str):
        vectors = await self.embed_texts_async([text])
        return vectors[0]
