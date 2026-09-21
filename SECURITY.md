# Security

Supported scope: maintained Qwen3.5-0.8B tooling on Python 3.12, saved-artifact
replay, final-paper numerical audits, and release packaging. The preserved
`src/sp_lense/steering/recorded_vector_fit.py` is inspection-only historical
source with original assertions/adapters. Native model inference is outside CI.

Treat repositories, Python code, models/lenses, and manifests as trusted local
inputs. A checksum detects change relative to a trusted manifest; it does not
authenticate an arbitrary third-party download. NumPy replay disables pickle.
Model loading disables remote code and uses configured revisions. The GPU notebook uses the recorded
model environment; cached reproduction does not require PyTorch.
Even `weights_only=True` is not a sandbox. Do not load unreviewed checkpoints or
run this toolkit with elevated privileges.

Release output directories must be controlled by the invoking user. Exclusive
writer locks and random temporary files prevent ordinary collisions; they do
not defend against an adversary who can replace parent directories or execute
code as the same user. Archives contain committed, selected public paths.
Review commits before release. Keep credentials in ignored local settings.

Do not post suspected credentials or exploit details in public issues. No private
reporting route has been verified for this repository. Open a minimal public
issue requesting a private contact without sensitive details; exchange details
only after the maintainer supplies a private route. Ordinary non-sensitive
defects may be filed in repository issues.

CI uses read-only tokens, pinned actions, static checks and dependency audits.
Full reachable-history/current-tree secret scans and research dependency
resolution are release checks; dated review evidence is in the wiki.
Checks cannot prove absence of vulnerabilities. Dependency reports are dated
observations; historical pins remain provenance, not a guarantee of safety.
