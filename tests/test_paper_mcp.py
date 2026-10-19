"""Paper identities, bounded file access and real MCP protocol checks."""
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from src.config import ROOT
from src.paper_mcp import PaperStore


class PaperScopeTests(unittest.TestCase):
    def test_markdown_preserves_the_clean_source(self):
        from src.paper_export import export
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'paper.md'
            report = export(destination=path)
            self.assertEqual((report['equations'], report['tables'], report['figures']), (4, 8, 2))
            self.assertEqual(path.read_bytes(), (ROOT/'paper/manuscript.md').read_bytes())

    def test_cohort_roles_and_unexecuted_results_remain_explicit(self):
        store = PaperStore()
        self.assertIn('fitting', store.get_cohort('H24')['data']['roles']['later_learned_corrections'])
        self.assertIsNone(store.get_cohort('R24')['data']['counts']['measured_successes'])
        self.assertEqual(store.get_result('small_model_screen')['data']['2B']['unexecuted_cases'], 12)
        self.assertFalse(store.get_result('steering_diagnostic')['data']['full_workflows_measured'])

    def test_changed_public_results_fail_before_serving(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ['paper/source_manifest.json', 'paper/results.json', 'paper/manuscript_submission.docx']+list(json.loads((ROOT/'paper/source_manifest.json').read_bytes())['agent_companion']['files'])
            for name in names:
                p = root/name; p.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/name, p)
            (root/'paper/results.json').write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                PaperStore(root)

    def test_replay_cannot_write_inside_code_or_immutable_evidence(self):
        with self.assertRaisesRegex(ValueError, 'outside the code'):
            PaperStore(output_root=ROOT/'outputs')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'outside immutable evidence'):
                PaperStore(artifacts=directory, output_root=Path(directory)/'outputs')

    def test_unknown_studies_and_arbitrary_replay_paths_are_rejected(self):
        store = PaperStore()
        with self.assertRaises(ValueError): store.get_cohort('Research4')
        with self.assertRaises(ValueError): store.read_replay('../.env')
        with self.assertRaises(ValueError): store.replay_results('arbitrary-command')


@unittest.skipUnless(importlib.util.find_spec('mcp'), 'Install requirements-mcp.txt for MCP protocol validation')
class ProtocolTests(unittest.TestCase):
    def test_real_stdio_protocol_from_an_unrelated_working_directory(self):
        import anyio
        from src.paper_mcp_validate import validate
        with tempfile.TemporaryDirectory() as directory:
            result = anyio.run(validate, Path(directory)/'report.json')
        self.assertEqual(result['predeclared_query_fixtures'], 22)
        self.assertEqual(len(result['tools']), 6)
        self.assertFalse(result['generative_agent_evaluated'])
        self.assertEqual(result['model_loads'], 0)

    def test_legacy_client_handshake_and_structured_output(self):
        import anyio
        from mcp import Client
        from src.paper_mcp import build_server
        async def check():
            async with Client(build_server(), mode='legacy') as client:
                value = await client.call_tool('get_result', {'analysis': 'workflow_comparison'})
                self.assertFalse(value.is_error)
                self.assertEqual(value.structured_content['data']['methods']['fine_tuned_A']['own']['successes'], 10)
                self.assertEqual(client.server_info.version, '1.0.0')
        anyio.run(check)
