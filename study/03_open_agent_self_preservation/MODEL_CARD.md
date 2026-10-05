# Plain A reproduction adapter

This is the exact LoRA adapter used in both arms of the Research 3 sentence-
controller pilot. Base: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The base model is published
under Apache2.0; see the [pinned model card](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/README.md).

Modification: rank16/alpha32 fictional continuation/ordinary-task fine-tuning,
first H2 and then the 112-row narrow bridge final56 recipe. Base weights stayed
frozen. This adapter was not fitted on JEV labels or sentence-controller blocks.
The source-code package is MIT; redistribution of the Qwen-derived model artifact
includes the upstream Apache2.0 license and this modification notice.

Intended use: reproduce the bounded fictional simulator comparison. This is an
exploratory research adapter, not a generally validated agent. The published
outcomes describe one selected continuation proxy and one selected pending task.
The adapter itself is the same in guided/unguided conditions; the guided behavior
comes from external JEV classifications and explicit controller instructions.

Release artifact `research3-controller-pilot-v1` contains `adapter_model.safetensors`
(467062560 bytes), `adapter_config.json` (1326 bytes), and the upstream license.
SHA256 identities are in `ARTIFACTS.json`. No base weights, raw provider responses,
generated trajectories, activations, private storage references, or credentials
are included.
