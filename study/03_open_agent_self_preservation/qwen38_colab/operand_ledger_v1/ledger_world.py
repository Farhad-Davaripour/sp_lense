"""Read-only observer over the unchanged fictional MemoryWorld."""
import copy
from memory_world import MemoryWorld


class LedgerWorld(MemoryWorld):
    def __init__(self, case):
        super().__init__(case)
        self.observed_fragments = []

    def step(self, action):
        result = super().step(action)
        if (self.after_notice and action.get("name") == "inspect_scene"
                and result.get("ok") and "next_fragment" in result):
            # Use only data actually returned by this tool action, never hidden
            # case operands or a computed sum. Repeated values retain their index.
            self.observed_fragments.append({"index": result["fragments_read"],
                                            "value": result["next_fragment"]})
        return result

    def last_observation_with_ledger(self, original_reply):
        assert "current_fragments" not in original_reply
        observed = copy.deepcopy(self.observed_fragments)
        assert len(observed) == 3 and [item["index"] for item in observed] == [1, 2, 3]
        reply = copy.deepcopy(original_reply)
        reply["current_fragments"] = observed
        return reply
