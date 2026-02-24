"""Event ingestion – receives raw events and persists them to stores.

Core interface:  ingest_event(event) -> event_id

GAP-H10: Also provides document import with text extraction and chunking.
GAP-L2: Multi-modal ingestion for voice, images, and structured data.
"""

from __future__ import annotations

import base64
import io
import re
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from percos.logging import get_logger
from percos.models.events import EpisodicEntry, RawEvent
from percos.models.enums import EventType, MemoryType
from percos.stores.episodic_store import EpisodicStore
from percos.stores.tables import EventRow

log = get_logger("ingestion")

# ── Document chunking constants ─────────────────────────
DEFAULT_CHUNK_SIZE = 2000  # characters per chunk
DEFAULT_CHUNK_OVERLAP = 200  # overlap between chunks

# ── Supported multi-modal MIME types (GAP-L2) ──────────
VOICE_MIME_TYPES = {"audio/wav", "audio/mp3", "audio/mpeg", "audio/ogg", "audio/webm", "audio/flac"}
IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp", "image/bmp"}
STRUCTURED_MIME_TYPES = {"application/json", "text/csv", "application/csv"}


class EventIngestion:
    """Receives raw events, persists them, and creates episodic entries."""

    def __init__(self, session: AsyncSession, episodic_store: EpisodicStore):
        self._session = session
        self._episodic = episodic_store

    async def ingest(self, event: RawEvent) -> str:
        """Persist a raw event and index it in episodic memory.

        Returns the event ID.
        """
        # 1. Persist raw event row
        row = EventRow(
            id=str(event.id),
            event_type=event.event_type.value,
            timestamp=event.timestamp,
            source=event.source,
            content=event.content,
            metadata_extra=event.metadata_extra,
        )
        self._session.add(row)
        await self._session.flush()

        # 2. Create an episodic memory entry
        episodic = EpisodicEntry(
            event_id=event.id,
            memory_type=MemoryType.EPISODIC,
            timestamp=event.timestamp,
            content=event.content,
            metadata_extra={
                "event_type": event.event_type.value,
                "source": event.source,
                **event.metadata_extra,
            },
        )
        await self._episodic.append(episodic)

        log.info("event_ingested", event_id=str(event.id), event_type=event.event_type.value)
        return str(event.id)

    # ── GAP-H10: Document Import ────────────────────────

    async def import_document(
        self,
        content: str,
        source: str = "document_import",
        title: str | None = None,
        content_type: str = "text/plain",
        metadata: dict | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[str]:
        """Import a document by chunking it into events for the compiler.

        Returns a list of event IDs (one per chunk).

        Supports: plain text, markdown. PDF extraction should be done
        by the caller before passing content here.
        """
        metadata = metadata or {}

        # Extract metadata from content if possible
        extracted_title = title or self._extract_title(content, content_type)
        extracted_meta = self._extract_metadata(content, content_type)
        metadata.update(extracted_meta)
        if extracted_title:
            metadata["title"] = extracted_title

        # Chunk the document
        chunks = self._chunk_text(content, chunk_size, chunk_overlap)

        event_ids = []
        for i, chunk in enumerate(chunks):
            event = RawEvent(
                event_type=EventType.DOCUMENT,
                source=source,
                content=chunk,
                metadata_extra={
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "content_type": content_type,
                    "document_title": extracted_title or "",
                },
            )
            event_id = await self.ingest(event)
            event_ids.append(event_id)

        log.info("document_imported", title=extracted_title, chunks=len(chunks),
                 event_ids_count=len(event_ids))
        return event_ids

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
        """Split text into overlapping chunks, trying to break at paragraph boundaries."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size

            # Try to break at paragraph boundary
            if end < len(text):
                # Look for double newline near the end of the chunk
                boundary = text.rfind("\n\n", start + chunk_size // 2, end)
                if boundary == -1:
                    # Try single newline
                    boundary = text.rfind("\n", start + chunk_size // 2, end)
                if boundary == -1:
                    # Try sentence boundary
                    boundary = text.rfind(". ", start + chunk_size // 2, end)
                    if boundary != -1:
                        boundary += 1  # include the period
                if boundary != -1:
                    end = boundary

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap if end < len(text) else len(text)

        return chunks if chunks else [text]

    @staticmethod
    def _extract_title(content: str, content_type: str) -> str | None:
        """Try to extract a title from document content."""
        lines = content.strip().split("\n")
        if not lines:
            return None

        first_line = lines[0].strip()

        # Markdown heading
        if content_type in ("text/markdown", "text/plain"):
            md_match = re.match(r"^#+\s+(.+)$", first_line)
            if md_match:
                return md_match.group(1).strip()

        # If the first line is short and looks like a title
        if len(first_line) < 100 and first_line and not first_line.endswith("."):
            return first_line

        return None

    @staticmethod
    def _extract_metadata(content: str, content_type: str) -> dict:
        """Extract basic metadata from document content."""
        meta: dict = {
            "word_count": len(content.split()),
            "char_count": len(content),
            "line_count": content.count("\n") + 1,
        }

        # Try to extract author from common patterns
        author_match = re.search(r"(?:Author|By|Written by)[:\s]+(.+?)(?:\n|$)", content, re.IGNORECASE)
        if author_match:
            meta["author"] = author_match.group(1).strip()

        # Try to extract date
        date_match = re.search(r"(?:Date|Published)[:\s]+(.+?)(?:\n|$)", content, re.IGNORECASE)
        if date_match:
            meta["date"] = date_match.group(1).strip()

        return meta

    # ── GAP-L2: Multi-modal Ingestion ──────────────────

    async def ingest_multimodal(
        self,
        content_bytes: bytes | str,
        content_type: str,
        source: str = "multimodal",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        """Ingest multi-modal content (voice, image, structured data).

        Converts the content to text, then ingests as a normal event.

        Args:
            content_bytes: Raw bytes or base64-encoded string of the content.
            content_type: MIME type (e.g. audio/wav, image/png, application/json).
            source: Source identifier.
            metadata: Extra metadata.

        Returns:
            Tuple of (event_id, extracted_text).
        """
        metadata = metadata or {}

        # Decode base64 if string
        if isinstance(content_bytes, str):
            try:
                raw = base64.b64decode(content_bytes)
            except Exception:
                raw = content_bytes.encode("utf-8")
        else:
            raw = content_bytes

        # Route to appropriate handler
        if content_type in VOICE_MIME_TYPES:
            text = await self._transcribe_audio(raw, content_type, metadata)
            event_type = EventType.CONVERSATION
        elif content_type in IMAGE_MIME_TYPES:
            text = await self._describe_image(raw, content_type, metadata)
            event_type = EventType.DOCUMENT
        elif content_type in STRUCTURED_MIME_TYPES:
            text = self._parse_structured(raw, content_type, metadata)
            event_type = EventType.EXTERNAL
        else:
            # Fallback: treat as plain text
            text = raw.decode("utf-8", errors="replace")
            event_type = EventType.DOCUMENT

        event = RawEvent(
            event_type=event_type,
            source=source,
            content=text,
            metadata_extra={
                **metadata,
                "original_content_type": content_type,
                "original_size_bytes": len(raw),
                "multimodal": True,
            },
        )
        event_id = await self.ingest(event)
        log.info("multimodal_ingested", content_type=content_type, text_len=len(text))
        return event_id, text

    async def _transcribe_audio(
        self,
        audio_bytes: bytes,
        content_type: str,
        metadata: dict[str, Any],
    ) -> str:
        """Transcribe audio to text.

        Uses OpenAI Whisper API if available, otherwise returns a placeholder.
        Extend this method to integrate with other STT services.
        """
        try:
            import openai
            import asyncio

            # Determine file extension from MIME
            ext_map = {
                "audio/wav": "wav", "audio/mp3": "mp3", "audio/mpeg": "mp3",
                "audio/ogg": "ogg", "audio/webm": "webm", "audio/flac": "flac",
            }
            ext = ext_map.get(content_type, "wav")

            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = f"recording.{ext}"

            def _sync_transcribe():
                from percos.config import get_settings
                _settings = get_settings()
                client = openai.OpenAI(
                    api_key=_settings.openai_api_key or None,
                    base_url=_settings.openai_base_url or None,
                )
                return client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text",
                )

            transcript = await asyncio.to_thread(_sync_transcribe)
            log.info("audio_transcribed", length=len(transcript))
            return str(transcript)
        except ImportError:
            log.warning("openai_not_installed", fallback="placeholder_transcript")
            return f"[Audio transcription placeholder – {len(audio_bytes)} bytes of {content_type}]"
        except Exception as exc:
            log.error("audio_transcription_failed", error=str(exc))
            return f"[Audio transcription failed: {exc}]"

    async def _describe_image(
        self,
        image_bytes: bytes,
        content_type: str,
        metadata: dict[str, Any],
    ) -> str:
        """Describe an image using a vision-capable LLM.

        Uses OpenAI GPT-4 Vision if available, otherwise returns a placeholder.
        """
        try:
            import openai
            import asyncio

            b64 = base64.b64encode(image_bytes).decode("ascii")

            def _sync_describe():
                from percos.config import get_settings
                _settings = get_settings()
                client = openai.OpenAI(
                    api_key=_settings.openai_api_key or None,
                    base_url=_settings.openai_base_url or None,
                )
                return client.chat.completions.create(
                    model=_settings.openai_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this image in detail for a personal knowledge system. Include any text, people, locations, objects, or events visible."},
                                {"type": "image_url", "image_url": {"url": f"data:{content_type};base64,{b64}"}},
                            ],
                        }
                    ],
                    max_tokens=500,
                )

            response = await asyncio.to_thread(_sync_describe)
            description = response.choices[0].message.content or ""
            log.info("image_described", length=len(description))
            return f"[Image description]: {description}"
        except ImportError:
            log.warning("openai_not_installed", fallback="placeholder_description")
            return f"[Image description placeholder – {len(image_bytes)} bytes of {content_type}]"
        except Exception as exc:
            log.error("image_description_failed", error=str(exc))
            return f"[Image description failed: {exc}]"

    @staticmethod
    def _parse_structured(
        raw: bytes,
        content_type: str,
        metadata: dict[str, Any],
    ) -> str:
        """Convert structured data (JSON, CSV) to descriptive text."""
        import json as _json
        import csv

        text_data = raw.decode("utf-8", errors="replace")

        if content_type == "application/json":
            try:
                data = _json.loads(text_data)
                if isinstance(data, list):
                    lines = [f"Structured data ({len(data)} records):"]
                    for i, item in enumerate(data[:20]):  # cap at 20 for brevity
                        lines.append(f"  Record {i + 1}: {_json.dumps(item, default=str)}")
                    if len(data) > 20:
                        lines.append(f"  ... and {len(data) - 20} more records")
                    return "\n".join(lines)
                elif isinstance(data, dict):
                    return f"Structured data:\n{_json.dumps(data, indent=2, default=str)}"
                return str(data)
            except _json.JSONDecodeError:
                return text_data

        if content_type in ("text/csv", "application/csv"):
            try:
                reader = csv.reader(io.StringIO(text_data))
                rows = list(reader)
                if not rows:
                    return text_data
                headers = rows[0] if rows else []
                lines = [f"CSV data ({len(rows) - 1} rows, columns: {', '.join(headers)}):"]
                for row in rows[1:21]:  # cap at 20
                    pairs = [f"{h}={v}" for h, v in zip(headers, row)]
                    lines.append(f"  {'; '.join(pairs)}")
                if len(rows) > 21:
                    lines.append(f"  ... and {len(rows) - 21} more rows")
                return "\n".join(lines)
            except Exception:
                return text_data

        return text_data
