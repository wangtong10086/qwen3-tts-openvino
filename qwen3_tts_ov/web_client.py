from __future__ import annotations

from functools import lru_cache
from importlib import resources


WEB_STATIC_PACKAGE = "qwen3_tts_ov.web_static"


def _read_web_resource(name: str) -> str:
    return resources.files(WEB_STATIC_PACKAGE).joinpath(name).read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def get_web_client_html() -> str:
    template = _read_web_resource("web_demo.html")
    css = _read_web_resource("web_demo.css")
    js = _read_web_resource("web_demo.js")
    return template.replace("{{WEB_DEMO_CSS}}", css).replace("{{WEB_DEMO_JS}}", js)


WEB_CLIENT_HTML = get_web_client_html()
