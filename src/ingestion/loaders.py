import os
from pathlib import Path
from typing import List, Dict, Any

import structlog
from docling.document_converter import DocumentConverter
from docling.chunking import HierarchicalChunker
from docling.datamodel.base_models import ConversionStatus

log = structlog.get_logger(__name__)

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".html", ".htm"})

converter = DocumentConverter()

def _convert(path: str) -> List[Dict[str, Any]]:
    try:
        result = converter.convert(path)
    except Exception as exc:
        log.error("docling_conversion_exception", path=path, error=str(exc))
        raise

    if result.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
        errors = [e.error_message for e in result.errors] if result.errors else []
        log.error(
            "docling_conversion_failed",
            path=path,
            status=result.status.value,
            errors=errors,
        )
        raise RuntimeError(f"Docling failed to convert {path!r}: {errors}")

    if result.status == ConversionStatus.PARTIAL_SUCCESS:
        log.warning("docling_partial_success", path=path)

    chunker = HierarchicalChunker()
    docling_chunks = chunker.chunk(result.document)

    filename = Path(path).name
    chunks_data = []

    for i, chunk in enumerate(docling_chunks):
        chunks_data.append(
            {
                "text": chunk.text,
                "source": filename,
                "chunk_index": i,
                "metadata": {
                    "filename": filename,
                    "headings": chunk.meta.headings
                    if hasattr(chunk.meta, "headings")
                    else [],
                },
            }
        )

    log.debug("docling_loaded_and_chunked", path=path, chunks_count=len(chunks_data))
    return chunks_data


def load_directory(directory: str) -> List[Dict[str, Any]]:
    all_chunks: List[Dict[str, Any]] = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if Path(fname).suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            fpath = os.path.join(root, fname)
            try:
                all_chunks.extend(_convert(fpath))
                log.info("loaded", path=fpath)
            except Exception as exc:
                log.warning("load_skipped", path=fpath, error=str(exc))
    return all_chunks


def load_files(paths: List[str]) -> List[Dict[str, Any]]:
    all_chunks: List[Dict[str, Any]] = []
    for path in paths:
        if Path(path).suffix.lower() not in SUPPORTED_EXTENSIONS:
            log.warning("unsupported_format", path=path)
            continue
        try:
            all_chunks.extend(_convert(path))
            log.info("loaded", path=path)
        except Exception as exc:
            log.warning("load_skipped", path=path, error=str(exc))
    return all_chunks
