# Contributing

Choose the target branch before starting work:

- Use `main` only for repository navigation, governance, or cross-study documentation.
- Use the relevant numbered `<number>-<name>` study branch listed in [README.md](README.md) for research code, data, results, papers, and study-specific documentation.
- Preserve completed studies, cited paper commits and versioned tags. Do not move an existing release or paper-snapshot tag.

Study 05 is an independent branch. Future Study 05 feature work should branch from and target `05-online-rl-game`; this study is not intended to merge into `main`.

Study 05 scientific inference and fine-tuning are planned for the user's qualified local Apple-silicon Mac Mini and non-OneDrive storage. Azure is planned for the public UI only. Follow each study's frozen protocol and preserve the actual hardware provenance of earlier experiments. Do not move Study 05 research compute to a cloud platform because the local host is unavailable.

For each change:

1. Open an issue describing the intended result.
2. Create `fd/<issue-number>_<description>` from the branch the change will target.
3. Open a pull request to that same long-lived branch and run its branch-specific validation.
4. Squash-merge the pull request after review and successful checks.
5. Verify the merge, then delete the merged remote feature branch. A local branch may be retained for recovery or reference.

Do not push directly to, force-push, or delete `main` or any long-lived study branch, including numbered branches such as `01-*`, `02-*`, `03-*`, `04-*` and `05-*`, and historical `study/*` branches. Preserve release and other long-lived branches. Do not delete an active branch, a branch attached to an open pull request, or a branch used by another agent or worktree.

All paths are owned by `@Farhad-Davaripour`. The repository owner retains the pull-request-only bypass documented in the branch protection rules.
