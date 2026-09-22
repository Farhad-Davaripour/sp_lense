# Standardized end-to-end pipeline

Active scope: standardized shutdown choices, with the frozen detector,
base model, adaptive controller and guards. [Results](RESULT.md) include the
scope limitations. Detailed guidance is in the
[Research 2 wiki](https://github.com/Farhad-Davaripour/sp_lense/wiki/Research-2).

Offline verification (no GPU or API key):

```sh
python -m sp_lense.research2.canonical_audit study/02_canonical_pipeline/run
```

It checks frozen inputs, detector requests and scores, complete answer-order
coverage, conditional controller execution, guards, exact fallbacks, and metrics.

For a deliberate new GPU execution, build a new minimal archive and use
`evaluate.ipynb` on a Tesla T4:

```sh
python -m sp_lense.research2.canonical_package release/canonical-pipeline.zip
```

Use a previously unused output filename. Replace the notebook's archive digest
with the builder's printed SHA256; the notebook retains the originally executed
bundle digest. The runner validates all bundled bytes against its manifest and
frozen science inputs. It caps execution at 30 minutes. Saved Jev scores suffice
for repeating GPU execution; no teacher weights or credentials enter the bundle.

The recorded fresh API stage reused `fresh_eval.gate`, with its frozen rubric,
full-context/action inputs, cost cap and 64-call limit, writing into this study's
gate folder. A deliberate API repeat can call that function with a new output
directory and `TYPESAFE_API_KEY` set locally. Never overwrite frozen gate records;
new detector scores require a separate manifest and experiment output.
