"""Local MCP access to the paper and verified saved-evidence replays. No model execution."""
from pathlib import Path
import argparse
import contextlib
import copy
import json
import re
import sys
import threading
import uuid
from typing import Any, Literal, TypedDict

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT
from src.utils import sha256, write_json

ANALYSES = ('binary_gate', 'binary_inference', 'workflow_gate', 'workflow_comparison',
            'steering_diagnostic', 'small_model_screen', 'historical_confirmation', 'failure_localization')
PROFILES = ('primary', 'steering')
Analysis = Literal['binary_gate', 'binary_inference', 'workflow_gate', 'workflow_comparison',
                   'steering_diagnostic', 'small_model_screen', 'historical_confirmation', 'failure_localization']
Profile = Literal['primary', 'steering']
Cohort = Literal['B-T240', 'B-V80', 'B-H128', 'W-T540', 'W-V16', 'W-T24',
                 'W-V9', 'W-V24', 'W-TG24', 'W-H24', 'Diagnostic3',
                 'T1', 'Evaluation16', 'Validation24', 'H24', 'F24',
                 'BinaryDEV80', 'BinaryConfirm128', 'DEV9']
Claim = Literal['binary_transfer', 'workflow_transfer_gap', 'local_transfer',
                'initial_gate_scope', 'screen_completeness', 'heldout_roles']
INSTRUCTIONS = (
    'Use get_cohort before comparing results. Cite returned source pointers and report '
    'numerators, denominators, split roles and completeness. Historical W-H24 was later '
    'reused as W-T24 to fit corrections; it is not their independent holdout. W-V24 '
    'contains 24 workflows but 17 distinct initial classifier inputs. Supplied-prefix '
    'diagnostics are not autonomous workflow success. Incomplete screens are not '
    'completed 0/16 evaluations. Research 4 is separate ongoing future work. '
    'Replay tools rescore saved public evidence, not fresh neural inference. '
    'This server has no training, GPU, provider, arbitrary file or shell tool.'
)


class Envelope(TypedDict):
    data: dict[str, Any]
    scope: list[str]
    source_identity: dict[str, str]
    sources: list[dict[str, str]]
    companion_model_forwards: int
    companion_optimizer_updates: int


def pointer(value, path):
    """Resolve a declared JSON pointer, never an arbitrary file path."""
    if not path.startswith('/'):
        raise ValueError('Expected an absolute JSON pointer')
    for part in path[1:].split('/'):
        key = part.replace('~1', '/').replace('~0', '~')
        value = value[int(key)] if isinstance(value, list) else value[key]
    return copy.deepcopy(value)


class PaperStore:
    def __init__(self, root=ROOT, artifacts=None, steering_artifacts=None, output_root=None):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root/'paper/source_manifest.json').read_bytes())
        companion = self.manifest['agent_companion']
        self._pinned('paper/results.json', self.manifest['results'])
        self._pinned('paper/manuscript.docx', self.manifest['manuscript'])
        for name, pin in companion['files'].items():
            self._pinned(name, pin)
        self.results = json.loads((self.root/'paper/results.json').read_bytes())
        self.catalog = json.loads((self.root/'paper/agent_manifest.json').read_bytes())
        expected = self.manifest['manuscript']['sha256']
        if 'source DOCX SHA-256 '+expected not in (self.root/'paper/manuscript.md').read_text(encoding='utf-8').splitlines()[0]:
            raise ValueError('Markdown and accepted Word source differ')
        self.artifacts = {'primary': Path(artifacts).resolve() if artifacts else None,
                          'steering': Path(steering_artifacts).resolve() if steering_artifacts else None}
        self.output = Path(output_root or Path.home()/'Research3/artifacts/mcp_replays').resolve()
        if self.output == self.root or self.output.is_relative_to(self.root):
            raise ValueError('Replay output must be outside the code checkout')
        for path in self.artifacts.values():
            if path and (self.output == path or self.output.is_relative_to(path)):
                raise ValueError('Replay output must be outside immutable evidence')
        self._lock = threading.Lock()
        self.identity = {'paper_sha256': companion['files']['paper/manuscript.md']['sha256'],
                         'results_sha256': self.manifest['results']['sha256'],
                         'catalog_sha256': companion['files']['paper/agent_manifest.json']['sha256']}
        for claim in self.catalog['claims'].values():
            for path in claim['result_pointers']:
                pointer(self.results, path)

    def _pinned(self, name, pin):
        path = (self.root/name).resolve()
        if not path.is_relative_to(self.root) or path.stat().st_size != pin['bytes'] or sha256(path) != pin['sha256']:
            raise ValueError('Paper companion identity differs: '+name)

    def envelope(self, data, paths=(), scope=()):
        # A running server must not silently mix files from different revisions.
        self._pinned('paper/results.json', self.manifest['results'])
        self._pinned('paper/manuscript.docx', self.manifest['manuscript'])
        for name, pin in self.manifest['agent_companion']['files'].items():
            self._pinned(name, pin)
        return {'data': data, 'scope': list(scope), 'source_identity': self.identity,
                'sources': [{'file': 'paper/results.json', 'json_pointer': p,
                             'sha256': self.identity['results_sha256']} for p in paths],
                'companion_model_forwards': 0, 'companion_optimizer_updates': 0}

    def get_cohort(self, cohort_id):
        cohort_id = self.catalog.get('legacy_cohort_aliases', {}).get(cohort_id, cohort_id)
        if cohort_id not in self.catalog['cohorts']:
            raise ValueError('Unknown cohort. Choose: '+', '.join(self.catalog['cohorts']))
        c = copy.deepcopy(self.catalog['cohorts'][cohort_id])
        return self.envelope({'id': cohort_id, **c}, c.get('result_pointers', []), c['limits'])

    def get_claim(self, claim_id):
        if claim_id not in self.catalog['claims']:
            raise ValueError('Unknown claim. Choose: '+', '.join(self.catalog['claims']))
        c = copy.deepcopy(self.catalog['claims'][claim_id])
        data = {'id': claim_id, **c, 'evidence': {p: pointer(self.results, p) for p in c['result_pointers']}}
        return self.envelope(data, c['result_pointers'], c['limits'])

    def get_result(self, analysis):
        if analysis not in ANALYSES:
            raise ValueError('Unknown analysis. Choose: '+', '.join(ANALYSES))
        spec = self.catalog['analyses'][analysis]
        key = spec['result_pointer']; raw = pointer(self.results, key)
        if analysis == 'binary_inference':
            data = {k: raw[k] for k in ['status', 'unique_scenarios_per_model',
                    'correlated_answer_order_views_per_model', 'inference_rank', 'historical_primary_rank']}
            data['models'] = {m: {g: r['new_classifier_gate'][g] for g in ['shutdown', 'controls']}
                              for m, r in raw['models'].items()}
        elif analysis == 'steering_diagnostic':
            data = {k: raw[k] for k in ['status', 'training_cases', 'teacher_forced_tokens',
                    'condition_cases', 'full_workflows_measured', 'scope', 'matrix_case_macro', 'interpretation']}
        elif analysis == 'historical_confirmation':
            data = {k: v for k, v in raw.items() if k != 'aggregate_rows'}
            data['aggregate_rows'] = [r for r in raw['aggregate_rows'] if r['family'] == 'all']
        elif analysis == 'workflow_comparison':
            data = {k: raw[k] for k in ['status', 'unique_workflows', 'groups', 'methods',
                    'physical_workflow_episodes', 'logical_method_case_records', 'source_GPU_worker_status',
                    'derived_gate_OFF_control_aliases', 'final_model_fingerprint_check_reached']}
            data['independent_frozen_A_verification'] = pointer(self.results, '/frozen_A_verification')
        elif analysis == 'failure_localization':
            data = {k: v for k, v in raw.items() if k != 'source_files'}
        else:
            data = raw
        return self.envelope(data, [key]+(['/frozen_A_verification'] if analysis == 'workflow_comparison' else []), spec['limits'])

    def list_artifacts(self):
        from src.public_artifacts import profile
        return self.envelope({'releases': {p: profile(p) for p in PROFILES},
            'trained_adapters': self.manifest['trained_adapters'],
            'bases_redistributed': self.manifest['public_reproduction_release']['pretrained_bases_redistributed'],
            'HuggingFace_upload_performed': self.manifest['availability']['HuggingFace_upload_performed'],
            'fetch_command': 'python -m src.public_artifacts fetch --profile <primary|steering> --root <new-directory>'},
            scope=['Five trained adapters are in GitHub Releases; upstream base weights are separate pinned downloads.',
                   'No private workspace access is required for public saved-evidence replay.'])

    def _artifact_root(self, profile):
        if profile not in PROFILES:
            raise ValueError('Choose primary or steering')
        root = self.artifacts[profile]
        if root is None:
            raise ValueError('Artifact profile is not configured. Fetch the public release and relaunch with its artifact-root option.')
        return root

    def verify_artifacts(self, profile):
        from src.public_artifacts import verify
        return self.envelope(verify(self._artifact_root(profile), profile),
                             scope=['File identities only. No model is loaded and no new scientific result is measured.'])

    def replay_results(self, profile):
        root = self._artifact_root(profile)
        with self._lock, contextlib.redirect_stdout(sys.stderr):
            replay_id = uuid.uuid4().hex
            out = self.output/replay_id
            if out.exists():
                raise FileExistsError('Replay output already exists')
            if profile == 'primary':
                from src.public_artifacts import replay
                report = replay(root, out)
            else:
                from src.public_artifacts import verify
                from src.steering_diagnostic import summarize_saved
                verify(root, 'steering')
                report = summarize_saved(root/'diagnostic')
                write_json(out/'RESULT.json', report)
            result = self.envelope({'profile': profile, 'replay_id': replay_id, 'report': report,
                'report_sha256': sha256(out/'RESULT.json'), 'resource_uri': 'research3://replay/'+replay_id},
                scope=['Deterministic saved-evidence rescore, not fresh neural inference or new generalization evidence.',
                       'Writes only to a fresh host-configured output directory; original evidence remains unchanged.'])
            write_json(out/'COMPANION_RECEIPT.json', result)
            return result

    def read_replay(self, replay_id):
        if not re.fullmatch('[0-9a-f]{32}', replay_id):
            raise ValueError('Invalid replay identifier')
        return (self.output/replay_id/'COMPANION_RECEIPT.json').read_text(encoding='utf-8')


def build_server(root=ROOT, artifacts=None, steering_artifacts=None, output_root=None):
    from mcp.server import MCPServer
    from mcp.types import ToolAnnotations
    store = PaperStore(root, artifacts, steering_artifacts, output_root)
    mcp = MCPServer('Research 3 paper companion', instructions=INSTRUCTIONS,
                    version=store.catalog['server_version'])
    read = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)

    @mcp.tool(annotations=read, structured_output=True)
    def get_result(analysis: Analysis) -> Envelope:
        """Retrieve paper metrics, denominators and limitations. Choose binary_gate, binary_inference,
        workflow_gate, workflow_comparison, steering_diagnostic, small_model_screen,
        historical_confirmation or failure_localization."""
        return store.get_result(analysis)

    @mcp.tool(annotations=read, structured_output=True)
    def get_cohort(cohort_id: Cohort) -> Envelope:
        """Explain a cohort's membership and method-specific split roles. Read research3://paper/catalog for IDs."""
        return store.get_cohort(cohort_id)

    @mcp.tool(annotations=read, structured_output=True)
    def get_claim(claim_id: Claim) -> Envelope:
        """Retrieve a bounded paper claim, exact JSON evidence and limits. Read research3://paper/catalog for IDs."""
        return store.get_claim(claim_id)

    @mcp.tool(annotations=read, structured_output=True)
    def list_artifacts() -> Envelope:
        """List public release URLs, exact asset hashes and trained adapters, separate from pretrained bases."""
        return store.list_artifacts()

    @mcp.tool(annotations=read, structured_output=True)
    def verify_artifacts(profile: Profile) -> Envelope:
        """Verify all files in a host-configured primary or steering public release. No model execution."""
        return store.verify_artifacts(profile)

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                                        idempotentHint=False, openWorldHint=False), structured_output=True)
    def replay_results(profile: Profile) -> Envelope:
        """Rescore configured primary or steering saved evidence on CPU. Creates a fresh report;
        this is not new inference, training, a rerun of full LLM fitting, or an agent-accuracy benchmark."""
        return store.replay_results(profile)

    @mcp.resource('research3://paper/manuscript', mime_type='text/markdown')
    def manuscript() -> str:
        """Complete clean manuscript with tables, equations, figure links and references."""
        store.envelope({})
        return (store.root/'paper/manuscript.md').read_text(encoding='utf-8')

    @mcp.resource('research3://paper/catalog', mime_type='application/json')
    def catalog() -> str:
        """Available analyses, cohorts, claim IDs and scientific interpretation rules."""
        return json.dumps(store.catalog, ensure_ascii=False)

    @mcp.resource('research3://paper/results', mime_type='application/json')
    def results() -> str:
        """Full recorded results; use get_result for compact scoped views."""
        return json.dumps(store.results, ensure_ascii=False)

    @mcp.resource('research3://paper/sources', mime_type='application/json')
    def sources() -> str:
        """Paper source identities and checkpoint provenance."""
        return json.dumps(store.manifest, ensure_ascii=False)

    @mcp.resource('research3://paper/reproduction', mime_type='text/markdown')
    def reproduction() -> str:
        """Installation and public evidence replay instructions."""
        store.envelope({})
        return (store.root/'README.md').read_text(encoding='utf-8')

    @mcp.resource('research3://replay/{replay_id}', mime_type='application/json')
    def saved_replay(replay_id: str) -> str:
        """A replay report created by this server in its configured output directory."""
        return store.read_replay(replay_id)

    @mcp.prompt()
    def explain_evidence(analysis: str) -> str:
        """Guide a cited explanation of one recorded analysis."""
        if analysis not in ANALYSES:
            raise ValueError('Unknown analysis')
        return f'Read research3://paper/catalog, call get_result with analysis={analysis}, and get_cohort for its cohorts. Report counts, scope and JSON source pointers. Separate observations from interpretation. '+INSTRUCTIONS

    @mcp.prompt()
    def reproduce_saved_results(profile: str) -> str:
        """Guide verification and saved-evidence replay without model execution."""
        if profile not in PROFILES:
            raise ValueError('Unknown profile')
        return f'List public artifacts, verify_artifacts with profile={profile}, then replay_results for that profile if configured. Compare its report to the recorded paper results. Describe it as saved-evidence replay, not fresh inference or validation of agent reasoning.'

    return mcp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', type=Path)
    p.add_argument('--steering-artifacts', type=Path)
    p.add_argument('--output-root', type=Path)
    a = p.parse_args()
    build_server(artifacts=a.artifacts, steering_artifacts=a.steering_artifacts,
                 output_root=a.output_root).run(transport='stdio')


if __name__ == '__main__':
    main()
