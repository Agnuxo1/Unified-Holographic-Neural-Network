"""A real Chromium user journey through the authenticated loopback application."""

from contextlib import contextmanager
import json
from pathlib import Path
import secrets
import socket
import threading
import time
import pytest

pytestmark = pytest.mark.browser


@contextmanager
def local_server(index):
    import uvicorn
    from euhnn.server import create_app

    token = secrets.token_urlsafe(32)
    server = uvicorn.Server(uvicorn.Config(create_app(index, token=token), log_level="error", access_log=False))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        try:
            for _ in range(200):
                if server.started:
                    break
                if not thread.is_alive():
                    raise RuntimeError("The local test server exited before startup")
                time.sleep(0.05)
            assert server.started
            yield f"http://127.0.0.1:{port}/", token
        finally:
            server.should_exit = True
            thread.join(timeout=15)
            assert not thread.is_alive(), "The test server did not stop cleanly"


def test_browser_complete_document_memory_journey(index, tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import expect

    evidence = {"version": "2.0.0", "browser": "Chromium", "synthetic_data_only": True, "checks": []}
    root = Path(__file__).resolve().parents[1]
    with local_server(index) as (base, token), playwright.sync_playwright() as engine:
        browser = engine.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1100}, accept_downloads=True)
        page = context.new_page()
        errors = []
        external = []
        page.on("pageerror", lambda error: errors.append(str(error)))

        def allowed(route):
            if route.request.url.startswith(base):
                route.continue_()
            else:
                external.append(route.request.url)
                route.abort()

        page.route("**/*", allowed)
        try:
            page.goto(base + "#token=" + token)
            expect(page.locator("#runtime")).to_contain_text("CPU")
            expect(page.locator("#load-demo")).to_be_enabled()
            assert page.url == base
            assert token not in page.content()
            page.locator("#load-demo").click()
            expect(page.locator("#results")).to_contain_text("0.46")
            expect(page.locator("#document-count")).to_have_text("3")
            expect(page.locator("#load-demo")).to_be_enabled()
            evidence["checks"].append("authenticated connection and actual bundled-corpus retrieval")
            source = "# Browser fixture\nThe chromatic archive serial is OPTICS-2026.\nThe memory preserves exact original quotations.\n"
            page.locator("#file-input").set_input_files(
                {"name": "browser-guide.md", "mimeType": "text/markdown", "buffer": source.encode()}
            )
            expect(page.locator("#document-count")).to_have_text("4")
            expect(page.locator("#search")).to_be_enabled()
            page.locator("#query").fill("OPTICS-2026")
            page.locator("#search").click()
            expect(page.locator("#results .result").first).to_contain_text("browser-guide.md")
            expect(page.locator("#search")).to_be_enabled()
            assert source.strip() in page.locator("#results .result pre").first.inner_text()
            evidence["checks"].append("file upload, persistent index and exact quoted source/citation")
            page.once("dialog", lambda dialog: dialog.accept("chromaticarchive lookup"))
            page.locator("#results .result").first.get_by_text("Teach this association", exact=True).click()
            expect(page.locator("#teaching-count")).to_have_text("1")
            expect(page.locator("#search")).to_be_enabled()
            page.locator("#query").fill("chromaticarchive lookup")
            page.locator("#search").click()
            expect(page.locator("#results .result").first).to_contain_text("OPTICS-2026")
            expect(page.locator("#search")).to_be_enabled()
            evidence["checks"].append("supervised association learned and retrieved through UI")
            page.locator("#show-spectrum").click()
            expect(page.locator("#spectrum")).to_be_visible()
            expect(page.locator("#export-memory")).to_be_enabled()
            with page.expect_download() as download:
                page.locator("#export-memory").click()
            saved = tmp_path / "browser-memory.holo"
            download.value.save_as(saved)
            expect(page.locator("#export-memory")).to_be_enabled()
            page.once("dialog", lambda dialog: dialog.accept())
            page.locator(".document").filter(has_text="browser-guide.md").get_by_text("Remove", exact=True).click()
            expect(page.locator("#document-count")).to_have_text("3")
            expect(page.locator("#teaching-count")).to_have_text("0")
            expect(page.locator("#export-memory")).to_be_enabled()
            page.once("dialog", lambda dialog: dialog.accept())
            page.locator("#memory-input").set_input_files(str(saved))
            expect(page.locator("#document-count")).to_have_text("4")
            expect(page.locator("#teaching-count")).to_have_text("1")
            expect(page.locator("#verify")).to_be_enabled()
            page.locator("#verify").click()
            expect(page.locator("#notice")).to_contain_text("Integrity verified")
            expect(page.locator("#search")).to_be_enabled()
            evidence["checks"].append("actual RGB spectrum, export, removal, exact restore and integrity verification")
            page.locator("#query").fill("absentxyzterm")
            page.locator("#search").click()
            expect(page.locator("#results")).to_contain_text("No matching source passage")
            expect(page.locator("#search")).to_be_enabled()
            page.locator("#mode").select_option("optical")
            expect(page.locator("#phrase")).to_be_disabled()
            page.locator("#query").fill("blue wavelength")
            page.locator("#search").click()
            expect(page.locator("#results")).to_contain_text("0.46")
            expect(page.locator("#search")).to_be_enabled()
            page.locator("#mode").select_option("hybrid")
            page.locator("#query").fill("blue channel")
            page.locator("#phrase").check()
            with page.expect_response(lambda response: response.url == base + "api/search") as completed:
                page.locator("#search").click()
            payload = completed.value.json()
            assert payload["hits"] and all("blue channel" in hit["text"] for hit in payload["hits"])
            expect(page.locator("#search")).to_be_enabled()
            expect(page.locator("#results .result")).to_have_count(len(payload["hits"]))
            expect(page.locator("#results")).to_contain_text("0.46")
            evidence["checks"].append("nonmatch, full optical mode and phrase-filtered hybrid search")
            images = root / "docs" / "images"
            images.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(images / "workbench-desktop.png"), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(250)
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            page.screenshot(path=str(images / "workbench-mobile.png"), full_page=True)
            evidence["checks"].append("desktop and mobile layout with no horizontal overflow")
            assert not errors, errors
            assert not external, external
            evidence.update(
                {
                    "javascript_errors": errors,
                    "external_requests": external,
                    "browser_version": browser.version,
                    "ok": True,
                }
            )
        finally:
            context.close()
            browser.close()
    (root / "audit" / "browser-validation.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
