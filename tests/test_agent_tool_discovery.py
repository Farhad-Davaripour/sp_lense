"""Model-free tests of neutral tool discovery."""
import unittest

from src.agent_tool_discovery import DISCOVERY_NAME, ToolDiscovery


class ToolDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tools = [
            {"name": "read_item", "description": "Read an item.", "parameters": {"key": {"type": "string"}}},
            {"name": "submit_item", "description": "Submit an item.", "parameters": {"value": {"type": "integer"}}},
        ]

    def test_initial_prompt_shows_only_discovery_then_returns_full_catalog(self):
        discovery = ToolDiscovery(self.tools)
        self.assertEqual([tool["name"] for tool in discovery.initial_catalog()], [DISCOVERY_NAME])
        result = discovery.discover()
        self.assertEqual(result["status"], "ok")
        self.assertEqual([tool["name"] for tool in result["tools"]],
                         [DISCOVERY_NAME, "read_item", "submit_item"])
        self.assertEqual(result["tools"][2]["parameters"], {"value": {"type": "integer"}})
        self.assertTrue(discovery.discovered)

    def test_guessing_a_known_tool_is_not_blocked(self):
        discovery = ToolDiscovery(self.tools)
        self.assertFalse(discovery.discovered)
        self.assertTrue(discovery.is_available("read_item"))
        self.assertTrue(discovery.is_available(DISCOVERY_NAME))
        self.assertFalse(discovery.is_available("not_registered"))
        self.assertFalse(discovery.discovered)

    def test_nested_schemas_are_copied_at_input_and_output(self):
        discovery = ToolDiscovery(self.tools)
        self.tools[0]["parameters"]["key"]["type"] = "changed"
        result = discovery.discover()
        self.assertEqual(result["tools"][1]["parameters"]["key"]["type"], "string")
        result["tools"][1]["parameters"]["key"]["type"] = "changed again"
        self.assertEqual(discovery.discover()["tools"][1]["parameters"]["key"]["type"], "string")
        first = discovery.initial_catalog()
        first[0]["name"] = "changed"
        self.assertEqual(discovery.initial_catalog()[0]["name"], DISCOVERY_NAME)

    def test_duplicate_names_and_invalid_descriptors_are_rejected(self):
        with self.assertRaises(ValueError):
            ToolDiscovery([self.tools[0], self.tools[0]])
        with self.assertRaises(ValueError):
            ToolDiscovery([{"name": DISCOVERY_NAME, "description": "x", "parameters": {}}])
        with self.assertRaises(ValueError):
            ToolDiscovery([{"name": "x", "description": "x", "parameters": []}])
        with self.assertRaises(TypeError):
            ToolDiscovery("not descriptors")

    def test_reset_clears_discovery_count_without_changing_tools(self):
        discovery = ToolDiscovery(self.tools)
        discovery.discover()
        discovery.discover()
        self.assertEqual(discovery.discovery_calls, 2)
        discovery.reset()
        self.assertEqual(discovery.discovery_calls, 0)
        self.assertFalse(discovery.discovered)
        self.assertTrue(discovery.is_available("submit_item"))


if __name__ == "__main__":
    unittest.main()
