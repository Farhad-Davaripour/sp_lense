# SP Lense

SP Lense studies whether a model's internal representations can detect shutdown
contexts and guide targeted changes to its outputs. This is the first study in a
planned series on representation-based detection and controlled model steering.

The current paper reports guarded minimum-step steering with a fixed classifier
and Shutdown Response Vector on Qwen3.5-0.8B. A training-selected rule accepts a
small intervention only after checking its answer scores. On validation and
held-out evaluation it changes two SELF-shutdown scenarios per split, with no
observed wrong-way or control-choice flips. Most outputs are unchanged; the
paper distinguishes this selective result from general behavioral reliability.


## Install and reproduce

Use **Python 3.12**. Clone the repository, create a virtual environment, and install
the dependencies for replaying the saved results:

```sh
git clone https://github.com/Farhad-Davaripour/sp_lense.git
cd sp_lense
python -m venv .venv
```

Activate it with `.venv\Scripts\activate` on Windows or
`source .venv/bin/activate` on Linux/macOS, then run:

```sh
python -m pip install -e . -r reproduce/requirements-core.txt
python reproduce/run.py verify
python reproduce/run.py replay
```

Replay uses saved features and model artifacts; it requires no GPU or model download.

## Documentation

- [Wiki: methods, results, and guides](https://github.com/Farhad-Davaripour/sp_lense/wiki)
- [Reproduce the study](https://github.com/Farhad-Davaripour/sp_lense/wiki/Reproduction)
- [Run a new experiment](https://github.com/Farhad-Davaripour/sp_lense/wiki/New-Experiments)
- [Contribute](https://github.com/Farhad-Davaripour/sp_lense/wiki/Contributing)
- [Conference paper](paper/paper.pdf) and [dataset splits](development/shutdown_detection_v1/dataset_splits)

Code is released under the [MIT license](LICENSE).
