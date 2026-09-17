"""Exercise the full local API and its authentication/input boundaries."""

from __future__ import annotations
import asyncio
import pytest
from fastapi.testclient import TestClient
from euhnn.server import create_app, LocalBoundary
from euhnn.hologram import decode_memory

TOKEN = "synthetic_test_token_" * 3


@pytest.fixture
def client(index):
    with TestClient(create_app(index, token=TOKEN), base_url="http://127.0.0.1:8765") as client:
        client.headers["Authorization"] = "Bearer " + TOKEN
        yield client


def test_static_workbench_has_no_token_and_sets_browser_boundaries(client):
    for path in ("/", "/app.js", "/style.css"):
        response = client.get(path)
        assert response.status_code == 200 and TOKEN not in response.text
        assert "no-store" in response.headers["cache-control"]
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
        assert response.headers["x-content-type-options"] == "nosniff"
    assert client.get("/.env").status_code == 404
    assert client.get("/api/openapi.json").status_code == 404


@pytest.mark.parametrize("path", ["/api/status", "/api/documents", "/api/verify", "/api/memory/export"])
def test_all_read_endpoints_require_authentication(client, path):
    assert client.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401


@pytest.mark.parametrize("host", ["evil.example", "127.0.0.1.evil.example", "user@127.0.0.1:8765", "localhost:999999"])
def test_host_rebinding_is_rejected(client, host):
    assert client.get("/api/status", headers={"Host": host}).status_code == 403


@pytest.mark.parametrize("origin", ["https://evil.example", "null", "http://127.0.0.1:8766", "https://127.0.0.1:8765"])
def test_cross_origin_is_denied_even_with_valid_token(client, origin):
    assert client.post("/api/demo", headers={"Origin": origin}).status_code == 403


def test_same_origin_is_allowed_without_wildcard_cors(client):
    response = client.get("/api/status", headers={"Origin": "http://127.0.0.1:8765"})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_full_upload_search_learn_memory_restore_and_delete(client, index):
    source = "# Original manual\nThe optical blue wavelength is 0.46.\nRailway code RAIL-742."
    response = client.put("/api/upload/manual.md", content=source.encode())
    assert response.status_code == 200
    doc_id = response.json()["document_id"]
    search = client.post("/api/search", json={"query": "blue wavelength"})
    hit = search.json()["hits"][0]
    assert hit["text"] == source[hit["start"] : hit["end"]]
    assert "upload/manual.md" in hit["citation"]
    assert client.post("/api/learn", json={"query": "ceruleanband", "chunk_id": hit["chunk_id"]}).status_code == 200
    assert client.post("/api/search", json={"query": "ceruleanband"}).json()["hits"][0]["learned_score"] > 0.9
    scene = client.post("/api/scene", json={"query": "blue wavelength"}).json()
    assert len(scene["rgb"]) == index.config.detector_count and scene["rt_cores_used"] is False
    memory = client.get("/api/memory/export")
    assert memory.status_code == 200 and memory.headers["content-disposition"].endswith('"euhnn-memory.holo"')
    assert decode_memory(memory.content)["documents"][0]["source"] == "upload/manual.md"
    assert client.get("/api/memory/preview").json()["raw_bytes"] > 0
    assert client.delete("/api/documents/" + doc_id).json()["deleted"] is True
    assert client.post("/api/search", json={"query": "blue"}).json()["hits"] == []
    assert client.put("/api/memory/import", content=memory.content).json()["documents"] == 1
    assert client.get("/api/verify").json()["ok"]
    assert client.post("/api/search", json={"query": "ceruleanband"}).json()["hits"]


def test_demo_load_is_real_and_idempotent(client):
    assert len(client.post("/api/demo").json()["documents"]) == 3
    assert all(d["unchanged"] for d in client.post("/api/demo").json()["documents"])
    assert client.post("/api/search", json={"query": "RAIL-742"}).json()["hits"]
    assert client.get("/api/status").json()["documents"] == 3


def test_pasted_document_and_text_only_injection(client):
    text = "Literal <script>alert(1)</script> optical instructions are source text, not HTML."
    assert client.post("/api/text", json={"source": "literal.txt", "text": text}).status_code == 200
    hits = client.post("/api/search", json={"query": "optical"}).json()["hits"]
    assert hits[0]["text"] == text
    # The frontend uses textContent; it never feeds a source result into innerHTML.
    javascript = client.get("/app.js").text
    assert "node.textContent = text" in javascript and ".innerHTML" not in javascript


@pytest.mark.parametrize(
    "payload",
    [
        {"query": None},
        {"query": 10},
        {"query": "q", "top_k": True},
        {"query": "q", "top_k": 0},
        {"query": "q", "unexpected": "path"},
        {"query": "x" * 4097},
    ],
)
def test_invalid_json_fields_fail_before_query(client, payload):
    assert client.post("/api/search", json=payload).status_code == 422


def test_unknown_query_mode_and_invalid_upload_are_not_successes(client):
    assert client.post("/api/search", json={"query": "x", "mode": "fake"}).status_code == 400
    assert client.put("/api/upload/bad.exe", content=b"invalid").status_code == 400
    assert client.put("/api/upload/a%5Cb.txt", content=b"invalid").status_code == 400
    assert client.post("/api/search", content=b"{").status_code == 422


def test_declared_body_limits_are_enforced(client):
    assert (
        client.post("/api/search", content=b"{}", headers={"Content-Length": str(3 * 1024 * 1024)}).status_code == 413
    )
    assert (
        client.put("/api/upload/test.txt", content=b"x", headers={"Content-Length": str(65 * 1024 * 1024)}).status_code
        == 413
    )
    assert client.post("/api/search", content=b"{}", headers={"Content-Encoding": "gzip"}).status_code == 400


def test_chunked_body_limit_and_authentication_precede_application_execution():
    async def scenario(authenticated):
        reached = False
        reads = 0
        sent = []

        async def app(scope, receive, send):
            nonlocal reached
            reached = True

        async def receive():
            nonlocal reads
            reads += 1
            return {"type": "http.request", "body": b"x" * (1024 * 1024), "more_body": True}

        async def send(message):
            sent.append(message)

        headers = [(b"host", b"127.0.0.1:8765")]
        if authenticated:
            headers.append((b"authorization", ("Bearer " + TOKEN).encode()))
        scope = {"type": "http", "method": "POST", "path": "/api/search", "scheme": "http", "headers": headers}
        await LocalBoundary(app, TOKEN)(scope, receive, send)
        assert not reached
        status = next(m["status"] for m in sent if m["type"] == "http.response.start")
        assert status == (413 if authenticated else 401)
        assert reads == (3 if authenticated else 0)

    asyncio.run(scenario(True))
    asyncio.run(scenario(False))
