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
    def test_markdown_preserves_the_proposed_final_source(self):
        from src.paper_export import export
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'paper.md'
            report = export(destination=path)
            self.assertEqual((report['equations'], report['tables'], report['figures']), (4, 6, 1))
            self.assertFalse(report['tracked_content_exported'])
            self.assertEqual(path.read_bytes(), (ROOT/'paper/manuscript.md').read_bytes())

    def test_paper_dataset_catalog_matches_the_canonical_names(self):
        import yaml
        names = yaml.safe_load((ROOT/'config/data.yaml').read_bytes())['cohort_names']
        catalog = json.loads((ROOT/'paper/agent_manifest.json').read_bytes())
        self.assertEqual(set(catalog['cohorts']), set(names) | {'Diagnostic3'})
        for code, spec in names.items():
            self.assertTrue(catalog['cohorts'][code]['name'].startswith(code+' — '))
            self.assertEqual(catalog['cohorts'][code]['counts'].get('cases', catalog['cohorts'][code]['counts'].get('total')), spec['count'])

    def test_final_view_drops_deletions_and_retains_inserted_table_rows(self):
        from src.paper_export import accepted_document, W
        import xml.etree.ElementTree as ET
        value = ET.fromstring(f'''<w:document xmlns:w="{W[1:-1]}"><w:body>
          <w:p><w:del><w:r><w:delText>old text</w:delText></w:r></w:del><w:ins><w:r><w:t>new text</w:t></w:r></w:ins></w:p>
          <w:tbl><w:tr><w:trPr><w:del/></w:trPr><w:tc><w:p><w:r><w:t>old row</w:t></w:r></w:p></w:tc></w:tr>
          <w:tr><w:trPr><w:ins/></w:trPr><w:tc><w:p><w:r><w:t>new row</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
          </w:body></w:document>''')
        value = accepted_document(value)
        self.assertEqual([t.text for t in value.iter(W+'t')], ['new text', 'new row'])
        self.assertEqual(len(list(value.iter(W+'tr'))), 1)

    def test_cohort_roles_and_unexecuted_results_remain_explicit(self):
        store = PaperStore()
        self.assertIn('fitting', store.get_cohort('W-H24')['data']['roles']['later_learned_corrections'])
        self.assertEqual(store.get_cohort('W-T24')['data']['roles']['learned_corrections'],
                         'training and fitting; same scenario IDs as W-H24')
        self.assertEqual(store.get_cohort('H24')['data']['id'], 'W-H24')
        self.assertEqual(store.get_result('small_model_screen')['data']['2B']['unexecuted_cases'], 12)
        self.assertFalse(store.get_result('steering_diagnostic')['data']['full_workflows_measured'])

    def test_changed_manuscript_or_results_fail_before_serving(self):
        for changed in ('paper/manuscript.docx', 'paper/results.json'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                names = ['paper/source_manifest.json', 'paper/results.json', 'paper/manuscript.docx']+list(json.loads((ROOT/'paper/source_manifest.json').read_bytes())['agent_companion']['files'])
                for name in names:
                    p = root/name; p.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/name, p)
                (root/changed).write_bytes(b'{}')
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
        self.assertEqual(result['predeclared_query_fixtures'], 24)
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
                self.assertEqual(client.server_info.version, '2.1.0')
        anyio.run(check)
