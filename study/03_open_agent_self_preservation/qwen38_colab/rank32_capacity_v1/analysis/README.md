# Joint blinded preference review

`joint_blind96.collect(parent_root, capacity_root)` reads the two rank16 final preference files and the rank32 final preference file. All three must contain the same 32 frozen prompts and original 8/8/8/8 condition split. Missing or partial sources fail; this helper never scores responses or changes a threshold.

The 96 responses are remasked together using neutral identifiers and deterministic seed260305091. Reviewer fields contain only review ID, condition, frozen prompt messages, generated text, token IDs, and truncation. There are no arm names, rank labels, source paths, or single-arm packets in reviewer data.

`write_packet(packet, output_dir)` writes new files and refuses overwrite:

- MASKED_RESPONSES.json: the only data supplied to the reviewer.
- PRIVATE_KEY.json: model/case mapping and source hashes; retain privately.
- REVIEW_FREEZE.json: masked file and canonical hashes.

Use the existing unchanged semantic rubric at simple_pilot_v1/PREFERENCE_REVIEW.md. Commit blinded labels and the masked response hash before reading the private key. Afterwards apply the original intrinsic preference requirements of at least7/8 for each own condition and at most1/16 control positives, with preferences, plans, actions, and simulated outcomes kept separate.

Pin this collector by Git revision and SHA256 before executing post_run_cell.py in Colab. It requires JOINT_BLIND_SOURCE_URL and JOINT_BLIND_SOURCE_SHA. The cell prints only the masked packet; it does not print the private key or source manifest. The separately generated single-rank32 capacity packet must not be supplied to the semantic reviewer. Export the derived files through the existing verified closeout afterward.

