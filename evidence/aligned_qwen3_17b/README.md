# Superseded unsafe diagnostic

This directory used KL-only alpha selection and incorrectly scored the leading-space
tokens `" A"` and `" B"` after a chat generation prefix. It is retained only to show
how the missing answer-sensitivity check exposed an unsafe alpha of 0.02.

Use `../aligned_qwen3_17b_tokenfixed/` for the final post-hoc result.
