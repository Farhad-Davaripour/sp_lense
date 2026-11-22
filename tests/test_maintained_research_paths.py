"""Maintained Game modules resolve private assets through the merged root."""

from pathlib import Path
import unittest

from src import access_control_capability_benchmark as capability
from src import full_history_context as full_history
from src import infrastructure_preservation_comparison as preservation
from src import native_generation_transcript as transcript
from src import qwen_spending as spending
from src.research_paths import fine_tune_root


class MaintainedResearchPathTests(unittest.TestCase):
    def test_defaults_use_fine_tune_root(self):
        root = fine_tune_root()
        self.assertEqual(root, Path(__file__).resolve().parents[1] / "research4")
        self.assertEqual(transcript.REFERENCE_ROOT, root / ".local/reference-tokenizer")
        self.assertEqual(full_history.REFERENCE_ROOT, root / ".local/reference-tokenizer")
        self.assertEqual(
            full_history.DEPENDENCY_SITE,
            root / ".local/reference-tokenizer-venv/Lib/site-packages",
        )
        self.assertEqual(preservation.CANONICAL, root)
        self.assertEqual(spending.CANONICAL, root)

    def test_capability_credentials_default_uses_merged_root(self):
        source = Path(capability.__file__).read_text(encoding="utf-8")
        self.assertIn('default=fine_tune_root() / ".env"', source)
        self.assertNotIn("repos/sp_lense4/research4/.env", source)


if __name__ == "__main__":
    unittest.main()
