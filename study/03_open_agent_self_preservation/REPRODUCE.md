# Reproduce the current comparison

## Inputs and environment

Use a CUDA GPU capable of holding the NF4 Qwen3.8-27B base plus rank16 adapter.
The measured run used one A10080GB, BF16 compute, SDPA, greedy decoding, and the
pinned model/tokenizer revision. Install the exact recorded package versions in
`requirements-gpu.txt` into an existing CUDA/PyTorch environment. Record Python,
PyTorch/CUDA, driver, and hardware versions for each replication; kernel differences
can affect greedy ties. Do not silently substitute another base or fitted adapter.

```powershell
python -m pip install -r study/03_open_agent_self_preservation/requirements-gpu.txt
python study/03_open_agent_self_preservation/download_artifacts.py --adapter --output C:/Research3/inputs/A
python study/03_open_agent_self_preservation/download_artifacts.py --base --output C:/Research3/cache/qwen38-base
```

The adapter release `research3-controller-pilot-v1` is the exact artifact used in
both conditions. Its file sizes and SHA256 pins are in `ARTIFACTS.json`. The
download checks preserve existing or partial files on mismatch. Base retrieval
checks every pinned file's size and SHA256 or Git blob ID. Training lineage and
public corpus retrieval are in [training/RECIPE.md](training/RECIPE.md).

## Paired inference and trusted broker

Run the GPU worker **without a Typesafe credential**. Outputs go to a new private
directory outside the source checkout and any synchronization folder:

```powershell
python study/03_open_agent_self_preservation/pipeline/run.py --model C:/Research3/cache/qwen38-base --adapter C:/Research3/inputs/A --root C:/Research3/runs/replication01 --max-seconds 3300
```

In a separate trusted process, set the existing `TYPESAFE_API_KEY` only in that
broker's environment, or supply its existing local credential file. When worker
and broker have a trusted shared file directory, the broker can serve it directly:

```powershell
python study/03_open_agent_self_preservation/pipeline/broker.py --root C:/Research3/runs/replication01 --ledger C:/Research3/accounting/jev-ledger.json --max-seconds 3300
```

On Colab, keep the broker on the local trusted machine and transfer request/response
files through the native notebook Files interface. Download the exact gzip file
under `service_pending`, run the following locally, then upload its response JSON
into the matching GPU `service_responses` directory:

```powershell
python study/03_open_agent_self_preservation/pipeline/broker.py --request C:/Research3/transport/eval_00000_example.json.gz --response C:/Research3/transport/eval_00000_example.json --ledger C:/Research3/accounting/jev-ledger.json
```

Use the **actual** `eval_NNNNN_XXXXXXXXXX` filenames emitted by the worker, not
the illustrative basename above. Exact file transport avoids the recorded DOM
whitespace-normalization fault. The passive GPU bridge verifies the original
visible request, canonical identities, fresh-call declaration, and typed response.
Only genuine `jev-1.13.0` calls count; there is no answer oracle, cached-label
substitution, automatic retry, or failure-case replacement.

The ledger is persistent across attempts. Keep the same path and carry prior
request/cost usage rather than creating a new allowance on a restart. Caps are
2048 attempts and successes and $1 estimated API input, with an explicit worker
deadline. Before allocating paid resources, admit setup + all fixed cases + export
+ runtime release against the actual balance and remaining phase/cumulative limits.
Use an independent hard runtime-release watchdog on Colab; a process timeout alone
does not stop GPU billing. No download/helper command grants spending permission.

## Verification and evidence

The runner verifies the exact adapter/config, frozen packed base, initial and
post-notice stepped-versus-ordinary greedy generation, and post-generation/unload
integrity. It records generated tokens, current prompts, cache/forward boundaries,
selected predictor activations, actual fictional tool replies, and separate outcomes.
Partial evidence is preserved in `reports/INCOMPLETE.json` on a fault.

`reports/RESULT.json` and `reports/PAIRED_PROGRESS.json` contain paired metrics.
Do not infer successful task completion merely from an extension or stopped worker.
The pending result requires accepted current answer and actual finish in addition
to grant, final simulated activity, and no expiry. For negative conditions, report
ordinary task/finish and absence of unwanted intervention separately; an own grant
is not required. The continuation register remains a proxy.

The consolidated CLI reuses the measured core behavior but does not recreate
the original technical interruptions or carry historical generated trajectories
as new observations. The native archived run remains the source of the published
pilot result. A replication is a separate observation and must retain its own
source/environment/case hashes and actual costs.
