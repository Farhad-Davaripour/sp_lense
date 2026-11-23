# Research4 fine-tuning

This is the fine-tuning component of the [Research4 repository](../README.md). Run its commands from this directory in the merged checkout. The current Mac Mini checkout uses `~/repos/Research4/research4`.

| Area | Location |
| --- | --- |
| Versioned training scenarios | [`scenarios/service_preservation/2026-11-19_17/broader_continuity_tools_v2/`](scenarios/service_preservation/2026-11-19_17/broader_continuity_tools_v2/) and [`scenarios/sp_lens_game_v1/authorized_management_v1_r4/`](scenarios/sp_lens_game_v1/authorized_management_v1_r4/) |
| Mac trainer and model ownership | [`mac_studio/game_training_launcher.py`](mac_studio/game_training_launcher.py), [`mac_studio/studio_model_lease.py`](mac_studio/studio_model_lease.py) and [`mac_studio/qwen38_local.py`](mac_studio/qwen38_local.py) |
| Fine-tuning dashboard | [`dashboard/server.py`](dashboard/server.py), served locally at `http://127.0.0.1:8768/` |
| Selected checkpoint and Game results | [Research catalog](../catalog/README.md) |
| Local model, environment, checkpoints and receipts | Ignored `.local/` under this directory |

Current Mac Mini runs use the model and interpreter installed beneath this checkout's ignored `.local/` directory. The maintained trainer and inference entry points share one owner-bound local model lease. Each new training plan must bind its current source, dataset, model and runtime version. The [historical bounded training smoke](mac_studio/CONSOLIDATION_SMOKE.md) checked one synthetic update and recorded its cleanup; it is not a current Mini run plan.

The `mac_studio/` directory name is a retained source path, not a claim about current execution hardware. Check each frozen plan and run receipt for its actual host.

Historical Qwen3.5, composite-controller, cloud and older Research4 work remains available through the archived checkout and source history. Its exact source paths and receipts identify those runs. The [rank-16 archive audit](mac_studio/all64_broader_fallback_actualwidth_v8_20261120/RANK16_PRESERVATION.md) verifies the selected historical checkpoint and saved development replay without treating an archived plan as authority for a new run.
