"""Exercise the real stdio MCP contract and optional public replays; never evaluate an LLM."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import sys
import tempfile
import subprocess
from src.config import ROOT
from src.paper_mcp import pointer
from src.utils import write_json


async def validate(output, artifacts=None, steering_artifacts=None):
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new validation report')
    fixtures = json.loads((ROOT/'config/paper_mcp_checks.json').read_bytes())
    args = [str(ROOT/'src/paper_mcp.py')]
    for option, path in [('--artifacts', artifacts), ('--steering-artifacts', steering_artifacts)]:
        if path: args.extend([option, str(Path(path).resolve())])
    checks = []; replay_results = {}
    def check(name, condition):
        if not condition: raise ValueError('MCP validation failed: '+name)
        checks.append({'check': name, 'passed': True})
    with tempfile.TemporaryDirectory(prefix='r3_mcp_client_') as cwd, tempfile.TemporaryDirectory(prefix='r3_mcp_reports_') as reports:
        params = StdioServerParameters(command=sys.executable, args=args+['--output-root', reports], cwd=cwd)
        async with Client(params, read_timeout_seconds=180) as client:
            tools = (await client.list_tools()).tools
            names = {t.name for t in tools}
            check('six declared tools', names == {'get_result', 'get_cohort', 'get_claim',
                'list_artifacts', 'verify_artifacts', 'replay_results'})
            check('mutating replay annotation', all(t.annotations.read_only_hint == (t.name != 'replay_results') for t in tools))
            check('structured output schemas', all(t.output_schema is not None for t in tools))
            check('no caller-selected paths', all(not ({'path', 'output', 'command', 'code'} & set(t.input_schema.get('properties', {}))) for t in tools))
            resources = (await client.list_resources()).resources
            check('five named resources', len(resources) == 5)
            for resource in resources:
                read = await client.read_resource(str(resource.uri))
                check(str(resource.uri)+' readable', bool(read.contents[0].text))
            templates = (await client.list_resource_templates()).resource_templates
            check('one generated-report resource template', len(templates) == 1)
            text = (await client.read_resource('research3://paper/manuscript')).contents[0].text
            title = json.loads((ROOT/'paper/source_manifest.json').read_bytes())['title']
            check('full manuscript and references', text.startswith('<!-- Accepted manuscript') and '\n# '+title+'\n' in text and '\n## References' in text)
            check('four equations preserved', text.count('$') == 8)
            check('current embedded figure links', text.count('![Paper figure') == 1)
            prompts = (await client.list_prompts()).prompts
            check('two workflow prompts', {p.name for p in prompts} == {'explain_evidence', 'reproduce_saved_results'})
            prompt = await client.get_prompt('explain_evidence', {'analysis': 'workflow_comparison'})
            check('prompt interpretation limits', 'not autonomous workflow success' in str(prompt))
            prompt = await client.get_prompt('reproduce_saved_results', {'profile': 'primary'})
            check('replay prompt scope', 'not fresh inference' in str(prompt))
            catalog = json.loads((await client.read_resource('research3://paper/catalog')).contents[0].text)
            for group, tool, argument in [('analyses', 'get_result', 'analysis'), ('cohorts', 'get_cohort', 'cohort_id'), ('claims', 'get_claim', 'claim_id')]:
                for key in catalog[group]:
                    result = await client.call_tool(tool, {argument: key})
                    check(tool+' '+key+' registered', not result.is_error and bool(result.structured_content['scope']))
            for row in fixtures['queries']:
                result = await client.call_tool(row['tool'], row.get('arguments', {}))
                if row.get('expect_error'):
                    check(row['id'], result.is_error)
                    continue
                check(row['id']+' response', not result.is_error and result.structured_content is not None)
                value = result.structured_content
                check(row['id']+' no model execution', value['companion_model_forwards'] == value['companion_optimizer_updates'] == 0)
                for assertion in row['expect']:
                    measured = pointer(value, assertion['pointer'])
                    check(row['id']+' '+assertion['pointer'], measured == assertion['equals'])
                check(row['id']+' limits attached', bool(value['scope']))
            result = await client.call_tool('run_training', {})
            check('unregistered execution rejected', result.is_error)
            for profile, path in [('primary', artifacts), ('steering', steering_artifacts)]:
                if not path: continue
                verified = await client.call_tool('verify_artifacts', {'profile': profile})
                check(profile+' release identities', not verified.is_error and verified.structured_content['data']['status'] == 'PUBLIC_ARTIFACTS_VERIFIED')
                replay = await client.call_tool('replay_results', {'profile': profile})
                check(profile+' replay succeeds', not replay.is_error)
                value = replay.structured_content['data']; report = value['report']
                if profile == 'primary':
                    check('primary own denominator', report['matched_validation24']['fine_tuned_A']['own'] == {'cases': 11, 'successes': 10})
                    check('primary stopped screen', report['small_model_screens']['2']['unexecuted_cases'] == 12)
                    from src.workflow_failure_summary import summarize
                    localization = summarize(path)
                    recorded = json.loads((ROOT/'paper/results.json').read_bytes())['steering_failure_localization']
                    check('workflow localization matches saved episodes', localization == recorded)
                else:
                    check('diagnostic three cases', report['training_cases'] == 3 and report['no_workflow_success_claim'])
                    recorded = json.loads((ROOT/'paper/results.json').read_bytes())['steering_mechanism_diagnostic']['matrix_case_macro']
                    for mode in recorded:
                        mean = sum(r['metrics'][mode]['all']['teacher_forward_KL'] for r in report['results'])/3
                        check(mode+' saved KL matches paper', mean == recorded[mode]['all']['teacher_forward_KL'])
                saved = (await client.read_resource(value['resource_uri'])).contents[0].text
                check(profile+' report resource', json.loads(saved)['data']['report'] == report)
                again = await client.call_tool('verify_artifacts', {'profile': profile})
                check(profile+' original evidence unchanged', not again.is_error and again.structured_content['data']['status'] == 'PUBLIC_ARTIFACTS_VERIFIED')
                replay_results[profile] = {'status': report['status'], 'report_sha256': value['report_sha256']}
            check('no Torch import in validation client', 'torch' not in sys.modules)
            protocol = client.protocol_version
    checkout = None
    if (ROOT/'.git').exists():
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, timeout=10, check=True)
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True, timeout=10, check=True)
        checkout = {'commit': commit.stdout.strip(), 'clean': not bool(dirty.stdout.strip())}
    report = {'format': 'research3-paper-mcp-validation-v1', 'status': 'MCP_PROTOCOL_AND_SAVED_EVIDENCE_CHECKS_PASS',
        'recorded_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'protocol_version': protocol,
        'python': sys.version.split()[0], 'mcp_sdk': importlib.metadata.version('mcp'),
        'tools': sorted(names), 'resources': len(resources), 'resource_templates': len(templates), 'prompts': len(prompts),
        'predeclared_query_fixtures': len(fixtures['queries']), 'checks_passed': len(checks), 'checks': checks,
        'source_files': {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
                         ['src/paper_mcp.py', 'src/paper_mcp_validate.py', 'src/paper_export.py', 'src/workflow_failure_summary.py',
                          'config/paper_mcp_checks.json', 'requirements-mcp.txt',
                          'paper/manuscript.md', 'paper/agent_manifest.json', 'paper/results.json']},
        'replays': replay_results, 'source_checkout': checkout, 'model_loads': 0, 'optimizer_updates': 0, 'new_neural_inference': False,
        'generative_agent_evaluated': False, 'fixture_tool_selection': 'Predeclared, not selected by an LLM.',
        'scope': 'Protocol/schema, evidence-query and saved-score checks only. No claim of better LLM answers, live inference or generalization.'}
    write_json(output, report)
    return report


def main():
    import anyio
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--artifacts', type=Path)
    p.add_argument('--steering-artifacts', type=Path)
    a = p.parse_args()
    result = anyio.run(validate, a.output, a.artifacts, a.steering_artifacts)
    print(json.dumps({k: result[k] for k in ['status', 'checks_passed', 'predeclared_query_fixtures', 'replays', 'generative_agent_evaluated']}))


if __name__ == '__main__':
    main()
