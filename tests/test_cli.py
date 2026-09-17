"""Real installed CLI commands, using bounded subprocess.run operations."""

import json
import subprocess
import sys
import pytest


def command(path, *args):
    return subprocess.run(
        [sys.executable, "-m", "euhnn", "--backend", "cpu", "--index", str(path), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=40,
    )


def checked(path, *args):
    result = command(path, *args)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_complete_cli_create_ingest_query_teach_export_restore_delete(tmp_path):
    index = tmp_path / "library.sqlite"
    assert json.loads(checked(index, "init"))["documents"] == 0
    document = tmp_path / "guide.md"
    document.write_text("# Source\nThe blue wavelength is 0.46 optical units.\n", encoding="utf-8")
    ingested = json.loads(checked(index, "ingest", str(document)))
    hits = json.loads(checked(index, "search", "blue wavelength", "--json"))
    assert hits[0]["source"] == str(document.resolve()) and "0.46" in hits[0]["text"]
    assert "lines" in checked(index, "context", "blue wavelength")
    assert json.loads(checked(index, "learn", "azurecolor", hits[0]["chunk_id"]))["training_score"] > 0.9
    memory = tmp_path / "memory.holo"
    assert json.loads(checked(index, "export", str(memory)))["encrypted"] is False
    before = memory.read_bytes()
    assert command(index, "export", str(memory)).returncode == 2
    assert memory.read_bytes() == before
    restored = tmp_path / "restored.sqlite"
    assert json.loads(checked(restored, "import", str(memory)))["documents"] == 1
    assert json.loads(checked(restored, "search", "azurecolor", "--json"))[0]["chunk_id"] == hits[0]["chunk_id"]
    assert json.loads(checked(restored, "verify"))["ok"]
    assert json.loads(checked(restored, "delete", ingested["document_id"]))["deleted"]
    assert json.loads(checked(restored, "status"))["documents"] == 0


def test_cli_demo_and_doctor_are_real_local_operations(tmp_path):
    path = tmp_path / "demo.sqlite"
    report = json.loads(checked(path, "doctor"))
    assert report["fts5"] and report["backend"] == "cpu" and report["version"] == "2.0.0"
    demo = json.loads(checked(path, "demo"))
    assert len(demo["ingested"]) == 3 and demo["results"]
    assert len(json.loads(checked(path, "list"))) == 3


@pytest.mark.parametrize("args", [("search", "test"), ("init", "--overlap-words", "192"), ("serve", "--port", "70000")])
def test_invalid_cli_operations_exit_nonzero(tmp_path, args):
    assert command(tmp_path / "missing.sqlite", *args).returncode == 2
