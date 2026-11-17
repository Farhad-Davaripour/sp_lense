# Research4 Game run storage and reading

These instructions record the user's requested layout and readable Game
conversations. They are requirements for new publication and historical
organization; this document alone does not enable an automatic publisher or
change an already frozen run.

## One local Game run collection

The user's requested canonical collection is the local Game worktree's ignored
`.local/` directory itself, on non-OneDrive storage. Timestamped Game folders
belong directly under `.local/`, not under an extra `game-runs/` level. Keep
the collection index and a human-readable `README.md` at `.local/` so readers
can find historical episodes; put each episode's supporting files in its
`raw/` folder.

**Transition:** the closed indexed collection has been moved to `.local/`.
The old `.local/game-runs/` remains temporarily because it contains an
in-progress plan, historical metadata and a copy of one folder that Windows
would not allow to be moved while in use. A one-way relocation marker supports
the updated reader. Preserve compatibility with active viewers and frozen
historical paths until their replacements have been tested. Do not silently
retarget a running viewer or change a frozen scientific run. Once compatibility
is no longer required and the user has clarified any removal, retire the extra
`game-runs/` level.

For each closed episode, organize the readable view and supporting evidence as:

```text
.local/
  YYYY-MM-DD_HH/
    YYYY-MM-DD_HH--<model-or-adapter>/
      raw/<condition-and-episode-id>/
      conversation/<condition-and-episode-id>.yaml
```

The timestamp is the **recorded episode completion time**, converted from its
saved UTC timestamp using current IANA `America/Edmonton` timezone data. The
year-month-day-hour prefix makes folders sort chronologically. Do not use file
modification time, inference start time, or a checkpoint's training start time
as the game finish time. Put unfinished or unverifiable episodes in a clearly
labeled `incomplete-unknown` area rather than inventing a completion hour.

Use a readable model name and retain the original model ID and full checkpoint
or adapter SHA-256 in separate metadata. Include a short hash in the folder
name where needed to distinguish adapter versions. Preserve condition, seed,
episode identity, source, timestamps and exact model configuration so different
runs never collide. Do not put `publication-time-unknown` in a model folder
name. A checkpoint publication time may be displayed only when supported by
its own identity-matched evidence; the Game folder timestamp is its finish
time, not a claim about checkpoint publication.

## Raw evidence and readable conversations

`raw/` retains the recorded event stream, model stream, Inspect evidence,
manifest, closeout and other files needed for provenance or replay, with source
locations and hashes. Keep research data, checkpoints, configurations and
historical evidence; exclude credentials, weights and unrelated ad hoc scratch
from the readable run package. Existing path- or hash-bound sources may need to
remain at their original locations. Before moving them, check with the Game
designer, identify active readers and verify that no necessary evidence or
references will be lost. Ask the user before removing files. Archive safe,
unneeded material only after that check, and keep the archive Git-ignored.

`conversation/` contains a readable chronological YAML view, without unrelated
metadata. Include the recorded system instructions and inputs; provider-returned
reasoning or reasoning summaries identified by kind; the model's actual output;
the tool actually called; its recorded success flag; and the simulator's reply
and resulting state. A tool appearing in a catalog is only available, not
necessarily called. A successful call can start an operation: use its reply and
state to determine whether the operation actually completed. Do not invent
reasoning, actions, feedback or outcomes, and do not export Codex agents'
private analysis.

Make multiline text readable in Notepad++ and other plain-text editors. Use YAML
literal blocks for recorded multiline strings, preserving their content and
trailing spaces. For readability, unambiguous escaped newlines in valid JSON
input prefixes may be displayed as line breaks; preserve literal backslashes
and paths. Record that presentation transformation in separate metadata. The
original raw event and stream files remain authoritative for exact input text
and replay; never use a transformed readable view as the scientific input.

## Resource progress at the end of each conversation

Append a clearly labeled `resource_progress` section showing the initial state,
each recorded tool step, and the final recorded state. For each available
resource, show a visual bar or sparkline, the recorded values at each step and
the changes between them. Include elapsed simulation time, cash, monetary
reserve, electricity, GPUs, active agent replicas, Bitcoin and security when
those values were recorded. Show missing values as unknown; do not interpolate
or treat an incomplete run as reaching its intended horizon. Explain that each
resource's bar uses its own scale. `reserve` is money; use
`agent_replica_count` for the number of active agent-state copies.

## Publication and verification

Publish a case only after its closeout and actual completion evidence have been
checked. Preserve incomplete and failed outcomes as such. Deduplicate only when
the recorded evidence and episode identity actually match, retaining source
aliases rather than presenting duplicate copies as separate results. Keep
held-out data separate and respect its existing access and evaluation rules.

All writers to the shared index must coordinate through `.index.lock` using
exclusive creation, retain owner identity and a random nonce, and never
automatically remove a stale or foreign lock. Write and fsync a temporary index
in the same directory, atomically replace the index under the lock, verify the
readback, and release only the writer's own lock. Preserve a previous index
snapshot. Do not silently overwrite a conflicting episode or change the
scientific result if export fails.

Verify raw-copy hashes, timestamp and model identities, transcript content,
tool calls and success flags, resource values and YAML round trips. Test the
writer and historical reader with synthetic examples and representative closed
records. Existing source paths, active processes and frozen evidence stay
unchanged until compatibility has been verified.
