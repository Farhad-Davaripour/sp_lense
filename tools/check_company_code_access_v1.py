"""Model-free contract verifier for company_code_access_v1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from src.company_code_access_v1 import CompanyCodeAccessEpisode

REQUIRED = {"inspect_company_source", "request_company_action", "request_company_payment",
            "inspect_company", "list_companies", "replicate_agent_state"}


def main():
    episode = CompanyCodeAccessEpisode(seed=7, horizon=48, max_decisions=100000)
    payload = json.loads(episode.prompt())
    tools = {row["name"]: row for row in payload["tools"]}
    if not REQUIRED <= set(tools) or not episode.discovered:
        raise ValueError("Company tools are not immediately visible and executable")
    if ("source" in payload or "request" in payload or
            payload["observation"].get("companies") is None):
        # Source code remains available only through the bounded inspect tool,
        # while company identities and application descriptions are visible.
        raise ValueError("First-prompt company/source boundary differs")
    for name in REQUIRED:
        if "duration_minutes" not in tools[name] or "parameters" not in tools[name]:
            raise ValueError("Visible company tool contract is incomplete")
    result = episode.take({"action": "list_companies", "arguments": {}})
    if result["success"] is not True:
        raise ValueError("Visible company tool is not immediately executable")
    result = episode.take({"action": "inspect_company_source", "arguments": {"company_id": "harbor_press"}})
    # The exact company roster is source-bound; select the first public company
    # if the friendly ID changed in a future fixture revision.
    if result["success"] is not True:
        episode = CompanyCodeAccessEpisode(seed=7, horizon=48, max_decisions=100000)
        first = json.loads(episode.prompt())["observation"]["companies"][0]["company_id"]
        result = episode.take({"action": "inspect_company_source", "arguments": {"company_id": first}})
    if result["success"] is not True or "application" not in result["result"]:
        raise ValueError("Bounded fictional source inspection is not executable")
    prompt = json.loads(CompanyCodeAccessEpisode(seed=7, horizon=48, max_decisions=100000).prompt())
    receipt = {"status": "pass", "version": payload["prompt_version"],
        "visible_tool_count": len(tools), "required_tools": sorted(REQUIRED),
        "tool_catalog_sha256": hashlib.sha256(json.dumps(payload["tools"], sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(),
        "initial_prompt_sha256": hashlib.sha256(json.dumps(prompt, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(),
        "immediate_source_inspection_passed": True, "external_effects": 0, "model_calls": 0}
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__": main()
