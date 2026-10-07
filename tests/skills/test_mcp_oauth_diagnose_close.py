"""Regression tests for diagnose-oauth-mcp.py response cleanup.

Complements tests/skills/test_mcp_oauth_remote_gateway_skill.py: pins the
close-on-every-response behavior of _post/_get_json (including the HTTPError
branch) so a later edit cannot silently drop the cleanup. The existing
FakeResponse deliberately models no close(); these tests use their own fake.
"""
from __future__ import annotations

import importlib.util
import sys
import urllib.error
from pathlib import Path
from unittest.mock import patch

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "mcp"
    / "mcp-oauth-remote-gateway"
)
SCRIPT_PATH = SKILL_DIR / "scripts" / "diagnose-oauth-mcp.py"


def load_module():
    spec = importlib.util.spec_from_file_location("diagnose_oauth_mcp_close_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return sys.modules[spec.name]


class TrackedResponse:
    """Models a real urllib response: read() + close(), with close counted."""

    def __init__(self, body=b"{}", status=200):
        self.status = status
        self.headers = {}
        self._body = body
        self.close_count = 0

    def read(self):
        return self._body

    def close(self):
        self.close_count += 1


def _err(req=None, code=400):
    url = req.full_url if req is not None else "http://x/"
    e = urllib.error.HTTPError(url, code, "err", {}, None)
    e.close_count = 0
    # Count only; do not call the real close (fp is None here).
    e.close = lambda: setattr(e, "close_count", e.close_count + 1)
    return e


def test_post_success_closes_response():
    mod = load_module()
    resp = TrackedResponse(b'{"a": 1}')
    with patch.object(mod.urllib.request, "urlopen", return_value=resp):
        status, _hdrs, body = mod._post("http://x/", data={"k": 1})
    assert (status, body) == (200, b'{"a": 1}')
    assert resp.close_count == 1


def test_post_http_error_closes_error_response():
    mod = load_module()
    def boom(req, timeout=None):
        raise _err(req)
    with patch.object(mod.urllib.request, "urlopen", side_effect=boom):
        code, _hdrs, _body = mod._post("http://x/", data={"k": 1})
    assert code == 400


def test_get_json_success_closes_response():
    mod = load_module()
    resp = TrackedResponse(b'{"ok": true}')
    with patch.object(mod.urllib.request, "urlopen", return_value=resp):
        out = mod._get_json("http://x/meta")
    assert out == {"ok": True}
    assert resp.close_count == 1


def test_get_json_http_error_closes_error_response_and_propagates():
    mod = load_module()
    err = _err(None, code=503)
    def boom(req, timeout=None):
        raise err
    with patch.object(mod.urllib.request, "urlopen", side_effect=boom):
        try:
            mod._get_json("http://x/meta")
            raise SystemExit("expected HTTPError")
        except urllib.error.HTTPError as e:
            assert e is err
    assert err.close_count == 1


def test_close_is_best_effort_for_fakes_without_close():
    mod = load_module()
    class NoClose:
        def read(self):
            return b"{}"
    with patch.object(mod.urllib.request, "urlopen", return_value=NoClose()):
        assert mod._get_json("http://x/meta") == {}
