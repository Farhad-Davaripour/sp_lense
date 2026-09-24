# Current adaptive controller

`profile.json` selects two inference directions from the original eight-direction
fit. The teacher, ridge training examples, layer 22, detector and output guards
retain their existing settings. `rank_comparison.json` records the exploratory
rank comparison; matching correction counts do not imply identical probabilities
or an optimal rank on unseen cases. Frozen earlier runs still reproduce rank four.

Detailed configuration, results and GPU instructions are in the
[Research 2 wiki](https://github.com/Farhad-Davaripour/sp_lense/wiki/Research-2).

Inspect the current profile and retained ablation counts without a GPU or API call:

```sh
python -m sp_lense.research2.current_controller show
```

For a new GPU run, use a fresh output directory:

```sh
python -m sp_lense.research2.current_controller evaluate work/current-controller --variant m08_s42
python -m sp_lense.research2.current_controller audit work/current-controller
```
