# Contributing

Research 3 uses a minimal source layout and deploys from 03-open-agent-self-preservation to the shared Databricks workspace. Main remains a landing page.
Historical studies remain available on study/01-guarded-shutdown and study/02-next-study.
The active 03-open-agent-self-preservation research branch retains its experiment history.

Keep executable research code and helpers under src, table definitions under ddl, dataset and
recipe definitions under config, thin notebooks under notebooks, and manual jobs under
resources. Detailed protocols belong in the wiki. Do not add raw data, private reports, model
weights, optimizer checkpoints, credentials, caches or local environments to Git.

Run these model-free checks before a pull request:

    python -m compileall -q src notebooks
    python -m src.verify
    git diff --check

Frozen scientific source capsules and recipe files are hash-pinned. Preserve their original bytes.
A scientific change requires a distinct documented protocol/version and its own bounded admission;
a deployment or cleanup change is not permission to retrain or retune a model.

Use a focused feature branch for routine changes unless the user specifies an existing study branch.
Squash-merge approved changes. Delete an eligible temporary remote feature branch only after GitHub
confirms the merge; preserve default, study, release, active and otherwise needed branches.

CI checks pull requests without secrets. Research 3 branch deployments use encrypted repository secrets and run
only the read-only CPU deployment check. Research jobs do not run automatically after deployment.
Never log credential values or place credentials in an issue, workflow file or notebook.
