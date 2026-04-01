import os
import re
from pathlib import Path

import structlog
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import ConversionStatus
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

log = structlog.get_logger(__name__)

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".html", ".htm"})

converter = DocumentConverter()
_header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")])
_char_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


def _clean_markdown(md: str) -> str:
    md = re.sub(r'<!--.*?-->', '', md, flags=re.DOTALL)
    md = re.sub(r'\.{4,}\s*\d*', '', md)
    md = re.sub(r'\n{3,}', '\n\n', md)
    return md.strip()


def _convert(path: str) -> list[dict[str, object]]:
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

    md = _clean_markdown(result.document.export_to_markdown())

    header_chunks = _header_splitter.split_text(md)
    final_chunks = _char_splitter.split_documents(header_chunks)

    filename = Path(path).name
    chunks_data = []

    for i, chunk in enumerate(final_chunks):
        chunks_data.append(
            {
                "text": chunk.page_content,
                "source": filename,
                "chunk_index": i,
                "metadata": {
                    "filename": filename,
                    "headings": [
                        chunk.metadata.get(h)
                        for h in ("h1", "h2", "h3")
                        if chunk.metadata.get(h)
                    ],
                },
            }
        )

    log.debug("docling_loaded_and_chunked", path=path, chunks_count=len(chunks_data))
    return chunks_data


def load_directory(directory: str) -> list[dict[str, object]]:
    all_chunks: list[dict[str, object]] = []
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


def load_files(paths: list[str]) -> list[dict[str, object]]:
    all_chunks: list[dict[str, object]] = []
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