from __future__ import annotations

import io
import os
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, Mock, patch

import pytest

from arelle.oim.Load import _preReadCsvFiles


class TestPreReadCsvFiles:
    """Tests for the _preReadCsvFiles optimization that pre-reads CSV file data from archives."""

    def _create_mock_model_xbrl(self, zip_path: str, file_contents: dict[str, bytes]) -> MagicMock:
        """Create a mock modelXbrl with a file source that reads from a ZIP."""
        mock = MagicMock()

        def mock_exists(path: str) -> bool:
            # Normalize path for comparison
            for key in file_contents:
                if path.endswith(key) or key in path:
                    return True
            return False

        def mock_file(path: str, binary: bool = False, **kwargs) -> tuple:
            for key, content in file_contents.items():
                if path.endswith(key) or key in path:
                    return (io.BytesIO(content),)
            raise FileNotFoundError(path)

        mock.fileSource.exists = mock_exists
        mock.fileSource.file = mock_file
        return mock

    def test_pre_read_csv_files_empty_tables(self) -> None:
        """Test with no tables returns empty dict."""
        mock_xbrl = MagicMock()
        result = _preReadCsvFiles(mock_xbrl, {}, "/tmp")
        assert result == {}

    def test_pre_read_csv_files_with_csv_tables(self) -> None:
        """Test that CSV files are pre-read into memory."""
        csv_content = b"col1,col2\nval1,val2\n"
        file_contents = {"table1.csv": csv_content}
        mock_xbrl = self._create_mock_model_xbrl("/tmp/test.zip", file_contents)

        tables = {
            "table1": {"url": "table1.csv"},
        }
        _dir = "/tmp"

        result = _preReadCsvFiles(mock_xbrl, tables, _dir)

        # Should have pre-read the CSV file
        assert len(result) == 1
        key = list(result.keys())[0]
        assert result[key] == csv_content

    def test_pre_read_csv_files_skips_xlsx(self) -> None:
        """Test that XLSX files are not pre-read (they use a different code path)."""
        mock_xbrl = MagicMock()
        mock_xbrl.fileSource.exists.return_value = True

        tables = {
            "table1": {"url": "table1.xlsx"},
        }
        _dir = "/tmp"

        result = _preReadCsvFiles(mock_xbrl, tables, _dir)
        assert len(result) == 0

    def test_pre_read_csv_files_skips_url_with_hash(self) -> None:
        """Test that URLs with hash fragments (workbook references) are not pre-read."""
        mock_xbrl = MagicMock()
        mock_xbrl.fileSource.exists.return_value = True

        tables = {
            "table1": {"url": "workbook.xlsx#Sheet1!"},
        }
        _dir = "/tmp"

        result = _preReadCsvFiles(mock_xbrl, tables, _dir)
        assert len(result) == 0

    def test_pre_read_csv_files_multiple_tables(self) -> None:
        """Test pre-reading multiple CSV files."""
        file_contents = {
            "table1.csv": b"a,b\n1,2\n",
            "table2.csv": b"c,d\n3,4\n",
            "table3.csv": b"e,f\n5,6\n",
        }
        mock_xbrl = self._create_mock_model_xbrl("/tmp/test.zip", file_contents)

        tables = {
            "t1": {"url": "table1.csv"},
            "t2": {"url": "table2.csv"},
            "t3": {"url": "table3.csv"},
        }
        _dir = "/tmp"

        result = _preReadCsvFiles(mock_xbrl, tables, _dir)
        assert len(result) == 3

    def test_pre_read_csv_files_nonexistent_file(self) -> None:
        """Test that non-existent files are gracefully skipped."""
        mock_xbrl = MagicMock()
        mock_xbrl.fileSource.exists.return_value = False

        tables = {
            "table1": {"url": "nonexistent.csv"},
        }
        _dir = "/tmp"

        result = _preReadCsvFiles(mock_xbrl, tables, _dir)
        assert len(result) == 0


class TestParallelValidation:
    """Tests for the parallel validation optimization in ValidateXbrl."""

    def test_thread_pool_executor_runs_tasks_concurrently(self) -> None:
        """Test that ThreadPoolExecutor can run multiple tasks and collect results."""
        import threading
        results = []
        lock = threading.Lock()

        def task(value: int) -> int:
            with lock:
                results.append(value)
            return value * 2

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(task, i) for i in range(3)]
            returned = [f.result() for f in futures]

        assert sorted(results) == [0, 1, 2]
        assert sorted(returned) == [0, 2, 4]

    def test_thread_pool_executor_propagates_exceptions(self) -> None:
        """Test that exceptions in ThreadPoolExecutor tasks are properly propagated."""
        def failing_task() -> None:
            raise ValueError("test error")

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(failing_task)
            with pytest.raises(ValueError, match="test error"):
                future.result()
