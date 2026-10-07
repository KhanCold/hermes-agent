"""Regression tests for google-workspace setup.py --revoke resource handling.

The fix under test: revoke() closes Google's HTTP response explicitly and no
longer makes a successful revoke depend on reading the response body
(Greptile finding: a read failure after Google accepted the request reported
failure while still deleting the local token).
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import types
import urllib.error
from pathlib import Path
from unittest.mock import patch

SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "productivity" / "google-workspace"
SCRIPT_PATH = SKILL_DIR / "scripts" / "setup.py"


def load_module():
    spec = importlib.util.spec_from_file_location("gws_setup_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return sys.modules[spec.name]


class _FakeCredentials:
    expired = False
    refresh_token = None
    token = "tok-1"

    @classmethod
    def from_authorized_user_file(cls, path, scopes=None):
        return cls()

    def refresh(self, request):
        raise AssertionError("refresh must not be called for non-expired creds")


def _install_fake_google():
    """revoke() imports google credentials inside the function; supply fakes."""
    fake_google = types.ModuleType("google")
    fake_oauth2 = types.ModuleType("google.oauth2")
    fake_creds_mod = types.ModuleType("google.oauth2.credentials")
    fake_creds_mod.Credentials = _FakeCredentials
    fake_auth = types.ModuleType("google.auth")
    fake_transport = types.ModuleType("google.auth.transport")
    fake_requests_mod = types.ModuleType("google.auth.transport.requests")
    fake_requests_mod.Request = type("Request", (), {})
    fake_google.oauth2 = fake_oauth2
    fake_oauth2.credentials = fake_creds_mod
    fake_google.auth = fake_auth
    fake_auth.transport = fake_transport
    fake_transport.requests = fake_requests_mod
    for name, mod in {
        "google": fake_google,
        "google.oauth2": fake_oauth2,
        "google.oauth2.credentials": fake_creds_mod,
        "google.auth": fake_auth,
        "google.auth.transport": fake_transport,
        "google.auth.transport.requests": fake_requests_mod,
    }.items():
        sys.modules[name] = mod


class _FakeResponse:
    """Models the real response contract used here: close() only.

    Deliberately has NO read() — a successful revoke must not depend on the
    body being readable (that was the bug).
    """

    def __init__(self):
        self.close_count = 0

    def close(self):
        self.close_count += 1


def _run_revoke(tmp_path, urlopen_impl):
    _install_fake_google()
    mod = load_module()
    home = tmp_path / "home"
    home.mkdir(parents=True)
    # Module-level paths are bound at import time; point them at the tmp home.
    mod.TOKEN_PATH = home / "google_token.json"
    mod.PENDING_AUTH_PATH = home / "google_oauth_pending.json"
    # Deps check requires the Hermes PM runtime; not under test here.
    mod._ensure_deps = lambda: None
    token = mod.TOKEN_PATH
    token.write_text("{}")

    buf = io.StringIO()
    # revoke() does `import urllib.request` inside the function, so it binds
    # the global module object at call time — patching urllib.request.urlopen
    # globally is exactly what a real urlopen caller would see.
    with patch.object(mod.os, "environ", dict(mod.os.environ, HERMES_HOME=str(home))), \
         patch("urllib.request.urlopen", side_effect=urlopen_impl), \
         contextlib.redirect_stdout(buf):
        mod.revoke()
    return buf.getvalue(), token


def test_revoke_success_closes_response_without_reading_body(tmp_path):
    resp = _FakeResponse()
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["timeout"] = timeout
        seen["url"] = req.full_url
        return resp

    out, token = _run_revoke(tmp_path, fake_urlopen)
    assert "Token revoked with Google." in out
    assert "Remote revocation failed" not in out
    assert resp.close_count == 1
    assert not token.exists()
    assert seen["timeout"] == 15
    assert "token=tok-1" in seen["url"]


def test_revoke_reports_failure_but_still_deletes_on_http_error(tmp_path):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, None)

    out, token = _run_revoke(tmp_path, fake_urlopen)
    assert "Remote revocation failed" in out
    assert not token.exists()
