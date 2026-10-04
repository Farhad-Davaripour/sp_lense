# Operational v3.1 accounting correction

The previous setup watchdog included prior compute cost by backdating the new allocation's setup clock. That mixed cost debt with elapsed setup time. The observed completion-attempt debit is0.82units, with no optimizer updates or active sessions after the stop.

These new modules preserve frozen v3. Prior debt is a separate phase_incurred_units field, default0.90 and never below the observed0.82. It is deducted from the same16-unit total; no budget is reset. At0.90,15.10future units remain and the cumulative ceiling stays148.92+16=164.92.

phase_started is the NEW current allocation start, including its current provision time only. Never subtract prior unit cost from that timestamp. Current setup_elapsed is now minus this start. The effective setup deadline is min(600seconds, remaining future budget after2700fit+3600evaluation+900export+0.5unit discretionary margin). At6.77 it is about9minutes23seconds. Check this plan BEFORE large downloads and monitor that current-only deadline.

Fit loading/audit/full56 updates retain2700seconds, full genuine-Jev B evaluation retains3600, and verified export/release retains900. The observed startup/56-update/final-audit estimate and all scientific/model/source/data/A-reuse gates are unchanged. Only the discretionary margin is0.5unit. An incomplete fit cannot be evaluated.

Overlay these admission.py and controller.py in the new control pack; retain the original byte-identical preflight.py and reuse_baseline.py. In config.phase use phase_incurred_units=0.90, margin_units=0.5, rate=the actually observed finite positive rate, and phase_started=the new allocation's monotonic start. The controller re-admits before any model child. No new study or model/API call was performed to prepare this correction.
