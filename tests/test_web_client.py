from __future__ import annotations

import re
from importlib import resources

from fastapi.testclient import TestClient

from qwen3_tts_ov import server
from qwen3_tts_ov.web_client import WEB_CLIENT_HTML, get_web_client_html


def test_web_demo_resources_are_packaged():
    root = resources.files("qwen3_tts_ov.web_static")

    for name in ("web_demo.html", "web_demo.css", "web_demo.js"):
        assert root.joinpath(name).is_file()


def test_web_client_html_inlines_packaged_resources():
    html = get_web_client_html()

    assert "{{WEB_DEMO_CSS}}" not in html
    assert "{{WEB_DEMO_JS}}" not in html
    assert 'id="modeButtons"' in html
    assert 'id="requestCount"' in html
    assert 'id="customRequestJson"' in html
    assert "runBackgroundRequest" in html
    assert WEB_CLIENT_HTML == html


def test_web_demo_js_element_ids_exist_in_html():
    root = resources.files("qwen3_tts_ov.web_static")
    html = root.joinpath("web_demo.html").read_text(encoding="utf-8")
    js = root.joinpath("web_demo.js").read_text(encoding="utf-8")
    html_ids = set(re.findall(r'id="([^"]+)"', html))
    js_ids = set(re.findall(r'\$\("([^"]+)"\)', js))

    missing = sorted(js_ids - html_ids)
    assert not missing


def test_web_routes_return_demo(monkeypatch, tmp_path):
    app = server.create_app(model_root=tmp_path / "openvino", warmup=False)
    client = TestClient(app)

    for path in ("/", "/web"):
        response = client.get(path)
        assert response.status_code == 200
        assert "Qwen3-TTS OpenVINO 控制台" in response.text
        assert "同时请求数" in response.text
