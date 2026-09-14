# Conference-paper draft

`manuscript.md` is the editable narrative. `manuscript.tex` is a portable LaTeX
version; `paper.pdf` is the rendered anonymous draft. This is a conference-style
draft, not a claim of acceptance or a venue-specific camera-ready template.

`build_figures.py` computes figure data from saved results and exports PNG/PDF
figures plus `data/figure_data.json`. `audit_claims.py` verifies counts, gates,
model predictions, and the source manifest. `build_paper.py` renders the paper.
Use `python reproduce/run.py figures`, `audit`, then `paper`.

See `venues.md` for checked publication routes and `submission_notes.md` for the
remaining author/venue decisions. The original empirical evidence is preserved;
no new training, steering or inference is performed by paper-building commands.

The study separates detector performance, conditional score shifts, preferred
label flips and task disturbance. The previously exposed holdout is diagnostic.
No fresh confirmatory results or reliable shutdown-control claim is made.
