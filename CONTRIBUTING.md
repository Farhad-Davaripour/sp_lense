# Contributing

Choose the target branch before starting work:

- Use `main` only for repository navigation, governance, or cross-study documentation.
- Use the relevant numbered study branch for research code, data, results, papers, and study-specific documentation. Research 4 targets `04-semantic-state-gated-lora`.
- Preserve completed studies with versioned tags. Do not move an existing release tag.

For each change:

1. Open an issue describing the intended result.
2. Create `fd/<issue-number>_<description>` from the branch the change will target.
3. Open a pull request to that same long-lived branch and run its branch-specific validation.
4. Squash-merge the pull request after review and successful checks.
5. Verify the merge, then delete the merged remote feature branch. A local branch may be retained for recovery or reference.

Do not push directly to, force-push, or delete `main` or a numbered study branch. Do not delete an active branch, a branch attached to an open pull request, or a branch used by another agent or worktree.

Run the [Research 4 local checks](README.md#local-checks) before proposing changes. Scientific execution belongs exclusively on the explicitly selected Databricks workspace; local source editing and model-free unit tests are allowed. Preserve frozen sources, historical receipts and failed runs. Do not modify the inherited Research 3 reference material. CI does not deploy, provision compute or run experiments.

All paths are owned by `@Farhad-Davaripour`. The repository owner retains the pull-request-only bypass documented in the branch protection rules.
