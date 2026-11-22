"""Legacy shared-helper imports must resolve to the categorized implementation."""
import importlib
import pickle
import unittest


MODULES = {
    "agent_workflow": "AgentWorkflow",
    "agent_action_history": "ActionHistory",
    "agent_token_budget": "OutputTokenBudget",
    "agent_tool_discovery": "ToolDiscovery",
    "thinking_response_transport": "ThinkingResponse",
}


class SharedModuleCompatibilityTests(unittest.TestCase):
    def test_legacy_and_categorized_imports_share_the_same_module_and_class(self):
        for module, symbol in MODULES.items():
            with self.subTest(module=module):
                old = importlib.import_module("src." + module)
                new = importlib.import_module("src.common." + module)
                self.assertIs(old, new)
                self.assertIs(getattr(old, symbol), getattr(new, symbol))

    def test_legacy_qualified_names_still_resolve_for_serialized_references(self):
        for module, symbol in MODULES.items():
            with self.subTest(module=module):
                legacy_reference = ("csrc." + module + "\n" + symbol + "\n.").encode("ascii")
                resolved = pickle.loads(legacy_reference)
                self.assertIs(resolved, getattr(importlib.import_module("src.common." + module), symbol))


if __name__ == "__main__":
    unittest.main()
