"""Regression tests for #22482 — prompt_toolkit print fallback."""

import unittest
from unittest.mock import MagicMock, patch


class TestPromptToolkitFallback(unittest.TestCase):
    """Verify cprint contains a try/except fallback to plain print."""

    def test_cprint_has_try_except(self):
        """cprint function body must contain try/except wrapping _pt_print."""
        import pathlib
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        banner_path = repo_root / "hermes_cli" / "banner.py"
        with open(banner_path) as f:
            source = f.read()
        # Find cprint function
        idx = source.find("def cprint(")
        self.assertGreater(idx, -1, "cprint not found")
        # Extract function body (next ~200 chars)
        block = source[idx:idx + 400]
        self.assertIn("try:", block, "Missing try block")
        self.assertIn("except", block, "Missing except block")
        self.assertIn("_pt_print", block, "Missing _pt_print call")
        self.assertIn("print(", block, "Missing fallback print")


if __name__ == "__main__":
    unittest.main()
