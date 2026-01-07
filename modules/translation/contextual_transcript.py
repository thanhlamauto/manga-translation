from typing import Any, List, Dict

import numpy as np

from .base import LLMTranslation
from ..utils.textblock import TextBlock
from processing import run_contextual_translation


class ContextualTranscriptLLMTranslation(LLMTranslation):
    """
    LLM-based translation engine that leverages the contextual transcript
    pipeline defined in `processing.py`.

    It converts the OCR'd TextBlock list into a transcript, runs the full
    context-aware pipeline (local context, character profiles, enriched
    transcript, translation), then maps the results back onto the blocks.
    """

    def __init__(self) -> None:
        self._settings: Any = None
        self._source_lang: str = ""
        self._target_lang: str = ""
        self._translator_key: str = ""

    def initialize(
        self,
        settings: Any,
        source_lang: str,
        target_lang: str,
        translator_key: str = "",
        **_: Any,
    ) -> None:
        self._settings = settings
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._translator_key = translator_key or "Context LLM (VN)"

    def translate(
        self,
        blk_list: List[TextBlock],
        image: np.ndarray,
        extra_context: str,
    ) -> List[TextBlock]:
        # Currently the pipeline is optimized for English -> Vietnamese.
        # We keep the signature generic so it can be extended later.
        if not blk_list:
            return blk_list

        transcript: List[Dict[str, Any]] = []
        for idx, blk in enumerate(blk_list):
            text = getattr(blk, "text", "") or ""
            text = text.strip()
            if not text:
                continue

            transcript.append(
                {
                    "index": idx,
                    "character": "Narrator",
                    "text": text,
                }
            )

        if not transcript:
            return blk_list

        try:
            result = run_contextual_translation(
                transcript,
                self._source_lang,
                self._target_lang,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[ERROR] ContextualTranscriptLLMTranslation failed: {exc}")
            return blk_list

        idx_to_vi = {item["index"]: item.get("text_vi", "") for item in result}

        for idx, blk in enumerate(blk_list):
            if idx in idx_to_vi:
                blk.translation = idx_to_vi[idx]

        return blk_list

