"""Persistence, exact citation, query safety and all-or-nothing update tests."""

from concurrent.futures import ThreadPoolExecutor
import sqlite3
import pytest
from euhnn import HolographicIndex, IndexConfig, TextPage, ValidationError, IndexFormatError
from euhnn.hologram import encode_memory, decode_memory

TEXT = """# Optical handbook
The blue wavelength is 0.46 simulation units. The green wavelength is 0.53.
Holographic memory stores phase and color while retaining exact source passages.
The red wavelength is 0.63 simulation units. Absorption reduces ray amplitude.
The coherent detector sums complex amplitudes and measures their intensity.
"""


def test_persistent_passages_keep_exact_offsets_and_lines(index):
    result = index.ingest_text(TEXT, source="handbook.md", metadata={"edition": 2})
    hits = index.search("blue wavelength")
    assert result["chunks"] >= 1 and hits
    for hit in hits:
        assert hit.text == TEXT[hit.start : hit.end]
        assert hit.line_start == 1 + TEXT.count("\n", 0, hit.start)
        assert hit.line_end == hit.line_start + hit.text.count("\n")
        assert hit.metadata == {"edition": 2}
        assert hit.source in hit.citation and hit.chunk_id in hit.citation
    path = index.path
    expected = [h.to_dict() for h in hits]
    index.close()
    with HolographicIndex(path, backend="cpu") as reopened:
        assert [h.to_dict() for h in reopened.search("blue wavelength")] == expected
        assert reopened.verify()["ok"]


def test_same_source_is_idempotent_and_changed_source_replaces_old_postings(index):
    a = index.ingest_text(TEXT, source="guide.md")
    b = index.ingest_text(TEXT, source="guide.md", title="Renamed", metadata={"a": 1})
    assert b["unchanged"] and b["document_id"] == a["document_id"]
    assert index.stats()["documents"] == 1
    assert index.documents()[0]["title"] == "Renamed"
    index.learn("optics", index.search("wavelength")[0].chunk_id)
    index.ingest_text("The railway replacement coupling is RAIL-742.", source="guide.md")
    assert not index.search("wavelength")
    assert index.search("RAIL-742")
    assert index.stats()["teaching"] == 0
    assert index.verify()["ok"]


def test_invalid_later_page_rolls_back_entire_replacement(index):
    index.ingest_text(TEXT, source="original")
    before = index.export_hologram()

    def pages():
        yield TextPage("replacement new text")
        raise ValidationError("The later page failed parsing")

    with pytest.raises(ValidationError):
        index.ingest_pages(pages(), source="original")
    assert index.export_hologram() == before
    assert index.search("wavelength") and not index.search("replacement")


def test_teaching_persists_and_does_not_claim_unsupervised_learning(index):
    index.ingest_text(TEXT, source="optics")
    target = index.search("blue wavelength")[0].chunk_id
    assert not index.search("azureband calibration procedure")
    evidence = index.learn("azureband calibration procedure", target)
    assert evidence["training_score"] > 0.9
    hit = index.search("azureband calibration procedure")[0]
    assert hit.chunk_id == target and hit.learned_score > 0.9
    with HolographicIndex(index.path, backend="cpu") as other:
        assert other.search("azureband calibration procedure")[0].chunk_id == target
    with pytest.raises(ValidationError):
        index.learn("the and of", target)
    with pytest.raises(ValidationError):
        index.learn("valid", "missing")


def test_external_writer_invalidates_cached_readout(index):
    index.ingest_text(TEXT, source="optics")
    index.search("blue")
    with HolographicIndex(index.path, backend="cpu") as other:
        other.learn("azurebeam retrieval", other.search("blue")[0].chunk_id)
    assert index.search("azurebeam retrieval")[0].learned_score > 0.9


def test_hologram_round_trip_preserves_source_and_teaching(index, tmp_path, config):
    index.ingest_pages([TextPage(TEXT, 7)], source="guide.pdf", metadata={"language": "en"})
    index.learn("cerulean retrieval", index.search("blue")[0].chunk_id)
    memory = index.export_hologram()
    with HolographicIndex(tmp_path / "restored.sqlite", create=True, config=config, backend="cpu") as other:
        evidence = other.import_hologram(memory)
        assert evidence["documents"] == 1 and evidence["teaching"] == 1
        assert other.export_hologram() == memory
        assert other.search("cerulean retrieval")[0].page == 7
        assert other.verify()["ok"]


def test_corrupt_import_does_not_modify_existing_memory(index):
    index.ingest_text(TEXT, source="guide")
    before = index.export_hologram()
    payload = decode_memory(before)
    payload["documents"][0]["chunks"][0]["end"] += 1
    with pytest.raises(ValidationError):
        index.import_hologram(encode_memory(payload))
    assert index.export_hologram() == before


def test_late_invalid_teaching_reference_rolls_back_import(index):
    index.ingest_text(TEXT, source="guide")
    before = index.export_hologram()
    payload = decode_memory(before)
    payload["teaching"] = [{"query": "unknown reference", "chunk_id": "missing"}]
    with pytest.raises(ValidationError):
        index.import_hologram(encode_memory(payload))
    assert index.export_hologram() == before


def test_delete_updates_all_storage_layers(index):
    doc = index.ingest_text(TEXT, source="guide")
    index.learn("blue color", index.search("blue")[0].chunk_id)
    assert index.delete_document(doc["document_id"])
    assert not index.delete_document(doc["document_id"])
    assert index.stats()["chunks"] == 0 and index.stats()["teaching"] == 0
    assert not index.search("blue")
    assert index.verify()["ok"]


def test_phrase_search_and_explicit_source_filter(index):
    index.ingest_text("The blue wavelength is 0.46. Other source material.", source="blue")
    index.ingest_text("Wavelength and a blue photon are related words in a different order.", source="other")
    assert [h.source for h in index.search("blue wavelength", phrase=True)] == ["blue"]
    assert all(h.source == "other" for h in index.search("blue", source="other"))
    with pytest.raises(ValidationError):
        index.search("blue", phrase=True, mode="optical")


@pytest.mark.parametrize("query", ["", "  ", "the and", "*", '" OR 1=1 --', "nonexistentxyzterm"])
def test_no_matching_lexical_evidence_has_no_invented_answer(index, query):
    index.ingest_text(TEXT, source="guide")
    assert index.search(query) == []


@pytest.mark.parametrize("mode", ["lexical", "hybrid", "optical"])
def test_query_modes_use_actual_passages(index, mode):
    index.ingest_text(TEXT, source="optical")
    index.ingest_text("Railway steam locomotive coupling maintenance procedure.", source="railway")
    hits = index.search("holographic memory phase color", mode=mode, top_k=1)
    assert len(hits) == 1 and hits[0].source == "optical"
    assert hits[0].text in TEXT


def test_bound_context_has_citations_and_never_exceeds_budget(index):
    index.ingest_text(TEXT * 10, source="long")
    context = index.context("wavelength", max_characters=256, top_k=5)
    assert len(context) <= 256 and "long:" in context


def test_two_connections_serialize_writes_without_losing_documents(index):
    def ingest(number):
        with HolographicIndex(index.path, backend="cpu") as own:
            own.ingest_text(f"Unique maintenance document record{number}.", source=f"source{number}")

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(ingest, range(12)))
    assert index.stats()["documents"] == 12
    assert index.verify()["verified_passages"] == 12


def test_corrupted_vector_is_reported_instead_of_returning_false_result(index):
    index.ingest_text(TEXT, source="guide")
    with sqlite3.connect(index.path) as db:
        db.execute("UPDATE chunks SET vector=?", (b"broken",))
    with pytest.raises(IndexFormatError):
        index.search("blue")
    with pytest.raises(IndexFormatError):
        index.verify()


def test_corrupted_source_and_lexical_postings_are_detected(index):
    index.ingest_text(TEXT, source="guide")
    with sqlite3.connect(index.path) as db:
        db.execute("UPDATE chunks SET text='invented content'")
    with pytest.raises(IndexFormatError):
        index.search("blue")
    with pytest.raises(IndexFormatError):
        index.export_hologram()
    with pytest.raises(IndexFormatError):
        index.verify()


def test_other_sqlite_database_is_never_overwritten(tmp_path):
    path = tmp_path / "other.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE important(value TEXT)")
    before = path.read_bytes()
    with pytest.raises(IndexFormatError):
        HolographicIndex(path, create=True, backend="cpu")
    assert path.read_bytes() == before


def test_missing_index_and_failed_creation_leave_no_broken_target(tmp_path, monkeypatch):
    path = tmp_path / "new.sqlite"
    with pytest.raises(ValidationError):
        HolographicIndex(path, backend="cpu")
    import euhnn.index as module

    def fail(*args, **kwargs):
        raise ValidationError("encoder failure")

    monkeypatch.setattr(module, "OpticalEncoder", fail)
    with pytest.raises(ValidationError):
        HolographicIndex(path, create=True, backend="cpu")
    assert not path.exists()


def test_reopening_with_incompatible_configuration_is_explicit(index):
    with pytest.raises(ValidationError):
        HolographicIndex(index.path, config=IndexConfig(), backend="cpu")


def test_database_path_with_unicode_spaces_and_query_characters(tmp_path, config):
    with HolographicIndex(
        tmp_path / "optica \u00f3 # library.sqlite", create=True, config=config, backend="cpu"
    ) as store:
        assert store.ingest_text("Unicode optical source.", source="\u00f3ptica")
        assert store.search("optical")[0].source == "\u00f3ptica"


def test_hybrid_preserves_strong_rare_identifier_evidence(index):
    """A lossy optical rank must not erase a decisive lexical relevance margin."""
    for number in range(48):
        index.ingest_text(
            f"The certified calibration pressure for MODULE{number:04d} is {number + 30} units. "
            "Pressure calibration inspection and maintenance documentation.",
            source=f"module-{number}",
        )
    for number in (0, 7, 17, 31, 47):
        query = f"certified calibration pressure MODULE{number:04d}"
        assert index.search(query, mode="lexical", top_k=1)[0].source == f"module-{number}"
        assert index.search(query, mode="hybrid", top_k=1)[0].source == f"module-{number}"


def test_stopword_only_phrase_keeps_exact_lexical_evidence(index):
    index.ingest_text("A literal quotation contains the and of in that order.", source="quote")
    hits = index.search("the and of", phrase=True)
    assert hits and hits[0].source == "quote"
    assert "the and of" in hits[0].text
