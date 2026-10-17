# SP Lense

SP Lense is a research program on detecting and steering model behavior through internal representations. The `main` branch is intentionally limited to navigation and repository governance. Research code, data, experiment records, and papers live on dedicated study branches.

## Studies

| Study | Status | Branch | Stable release |
| --- | --- | --- | --- |
| Study 01 Guarded shutdown steering | Completed | [`study/01-guarded-shutdown`](https://github.com/Farhad-Davaripour/sp_lense/tree/study/01-guarded-shutdown) | [`study-01-v1.0.0`](https://github.com/Farhad-Davaripour/sp_lense/releases/tag/study-01-v1.0.0) |
| Study 02 Adaptive activation steering | Active | [`study/02-next-study`](https://github.com/Farhad-Davaripour/sp_lense/tree/study/02-next-study) | Not released |

To inspect or reproduce a study, switch to its branch and follow that branch's README. The completed Study 01 [paper](https://github.com/Farhad-Davaripour/sp_lense/blob/study/01-guarded-shutdown/paper/paper.pdf) and reproducibility package remain on the Study 01 branch and release tag.

## Branch model

- `main` is the protected landing and governance branch.
- `study/<number>-<name>` branches contain research-specific work and retained study history.
- `fd/<issue-number>_<description>` branches are short-lived feature branches. They target the relevant long-lived branch, are squash-merged, and are deleted from the remote after a successful merge.

Repository-wide documentation is maintained in the [wiki](https://github.com/Farhad-Davaripour/sp_lense/wiki). Proposed work and review history are tracked in [issues](https://github.com/Farhad-Davaripour/sp_lense/issues) and [pull requests](https://github.com/Farhad-Davaripour/sp_lense/pulls).

Code and documentation are released under the [MIT license](LICENSE).
