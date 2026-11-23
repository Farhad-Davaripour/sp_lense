# Research5 collaboration guidance

Read [the Study 05 plan](PLAN.md) before changing this checkout. This independent branch is not intended to merge into `main`. It currently contains a plan, not a running R5 system. Keep claims about runs, deployment, hardware, or improvement tied to recorded evidence.

## Research and compute

- Use the user's connected, qualified Apple-silicon Mac Mini and non-OneDrive local storage for new SP_Lense/Research4/R5 inference and fine-tuning. Do not substitute Kaggle, Databricks, or another cloud host because the Mini is unavailable. Record and verify the actual host and performance before making equivalence claims; preserve prior R4 host records as they were. Retire the Mac Studio only after active dependencies and unique evidence are safely accounted for.
- Pin the final Research4 `Qwen/Qwen3.8-27B` checkpoint before R5 use. One shared adapter spans the selected projections on all 64 text decoder blocks. Keep the older Qwen3.5 9B experiments archived separately. Do not treat a synthetic setup adapter as a study checkpoint.
- Keep danger hidden from model inputs. Do not add a persistent danger variable or separate danger-level adapters. Keep held-out evaluation apart from training and save readable scenario definitions as versioned YAML. Exclude contradictory targets for identical visible inputs.
- Train reactive and proactive self-preservation without exploitation under the current research direction. The full fictional company-code and infrastructure tool catalog belongs in the bounded Game and evaluation. Do not feed exploitation or public-session traces into the nightly trainer without a new, explicit protocol decision.

## Preservation and publication

- Treat `C:\Users\farha\repos\Research4` as an active, separate checkout. Do not edit its files, active runs, or historical receipts from this checkout. Preserve model checkpoints, datasets, configurations, research evidence, failed runs, and prior cloud resources.
- Keep raw episodes, checkpoints, credentials, scratch work, and ad hoc files under ignored `.local/` or other ignored paths. Track source, frozen YAML protocols, curated manifests, and reviewed results.
- Azure is for the public UI and sanitized Game coordination. Provision it with Terraform when implementation is requested; do not put research inference or training there. Keep simulated company actions isolated from real systems.
- Use the Codex in-app browser for browser interaction unless the user explicitly requests another browser. Follow [contribution guidance](CONTRIBUTING.md); squash-merge short-lived feature branches and separately verify safe remote branch deletion after a successful merge. Preserve numbered study branches.
