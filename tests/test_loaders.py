import pytest
from unittest.mock import patch, MagicMock
from src.ingestion.loaders import load_directory, load_files, SUPPORTED_EXTENSIONS


def _make_mock_chunks(filename: str, n: int = 2):
    return [
        {"text": f"Chunk {i} from {filename}", "source": filename, "chunk_index": i,
         "metadata": {"filename": filename, "headings": []}}
        for i in range(n)
    ]


def test_supported_extensions_contains_pdf_html():
    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert ".html" in SUPPORTED_EXTENSIONS
    assert ".htm" in SUPPORTED_EXTENSIONS


def test_load_directory_skips_unsupported_files(tmp_path):
    (tmp_path / "file.txt").write_text("ignored")
    (tmp_path / "file.csv").write_text("ignored")
    with patch("src.ingestion.loaders._convert") as mock_convert:
        result = load_directory(str(tmp_path))
    mock_convert.assert_not_called()
    assert result == []


def test_load_directory_calls_convert_for_each_supported_file(tmp_path):
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "doc.html").write_text("<html></html>")

    with patch("src.ingestion.loaders._convert") as mock_convert:
        mock_convert.side_effect = lambda p: _make_mock_chunks(p)
        result = load_directory(str(tmp_path))

    assert mock_convert.call_count == 2
    assert len(result) == 4  # 2 files × 2 chunks each


def test_load_directory_skips_failed_files(tmp_path):
    (tmp_path / "good.pdf").write_bytes(b"%PDF fake")
    (tmp_path / "bad.pdf").write_bytes(b"%PDF fake")

    def side_effect(path):
        if "bad" in path:
            raise RuntimeError("conversion failed")
        return _make_mock_chunks(path)

    with patch("src.ingestion.loaders._convert", side_effect=side_effect):
        result = load_directory(str(tmp_path))

    assert len(result) == 2  # only good.pdf chunks


def test_load_directory_empty_dir_returns_empty(tmp_path):
    result = load_directory(str(tmp_path))
    assert result == []


def test_load_files_skips_unsupported():
    with patch("src.ingestion.loaders._convert") as mock_convert:
        result = load_files(["file.txt", "data.csv"])
    mock_convert.assert_not_called()
    assert result == []


def test_load_files_processes_supported():
    with patch("src.ingestion.loaders._convert") as mock_convert:
        mock_convert.return_value = _make_mock_chunks("doc.pdf")
        result = load_files(["doc.pdf"])

    assert len(result) == 2
    assert all(c["source"] == "doc.pdf" for c in result)
