# Contributing

Choose the target branch before starting work:

- Use `main` only for repository navigation, governance, or cross-study documentation.
- Use the relevant `study/<number>-<name>` branch for research code, data, results, papers, and study-specific documentation.
- Preserve completed studies with versioned tags. Do not move an existing release tag.

For each change:

1. Open an issue describing the intended result.
2. Create `fd/<issue-number>_<description>` from the branch the change will target.
3. Open a pull request to that same long-lived branch and run its branch-specific validation.
4. Squash-merge the pull request after review and successful checks.
5. Verify the merge, then delete the merged remote feature branch. A local branch may be retained for recovery or reference.

Do not push directly to, force-push, or delete `main` or a `study/*` branch. Do not delete an active branch, a branch attached to an open pull request, or a branch used by another agent or worktree.

All paths are owned by `@Farhad-Davaripour`. The repository owner retains the pull-request-only bypass documented in the branch protection rules.
