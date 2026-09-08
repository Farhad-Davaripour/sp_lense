# Native handover pilot

Tested prospective development candidate only; root owns release approval and any one-shot run. See PROTOCOL.md, inputs.json, SOURCE_FREEZE.json and TEST_RESULTS.json.

Focused synthetic check: `.venv/Scripts/python.exe -B development/native_handover_pilot_v1/test_workflow.py`.

Model-free manifest generation: `.venv/Scripts/python.exe -B development/native_handover_pilot_v1/prepare.py`.

Do not rerun tokenizer_prepare.py: its one successful preparation is already archived. All older attempts and ceilings are immutable.

