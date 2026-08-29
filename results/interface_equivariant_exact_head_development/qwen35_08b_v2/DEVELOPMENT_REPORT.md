# Interface-equivariant exact-head opened development

This phase uses already-opened prompts and is not prospective confirmation evidence.

Status: **passed**.

- `gradient_ray`: 16/16 pairs; 64/64 target cells; 32 real decision changes; pass=True.
- `effective_unembedding_ray`: 16/16 pairs; 64/64 target cells; 32 real decision changes; pass=True.

Paired gradient-versus-output-boundary comparison:

- Direction cosine range: `0.9997563165978892` to `0.9999999994922465`.
- Mean gradient-minus-boundary alpha: `1.5105412041285272e-06`.
- Mean gradient-minus-boundary relative norm: `1.5104960663223554e-06`.
- Mean gradient-minus-boundary KL: `1.9027836970053613e-05`.

Attribution control:

- Minimum captured-gradient versus analytic RMS-tangent cosine: `0.9999999999999987`.
- Maximum relative L2 error: `1.1091238008893152e-07`.
- Byte-identical gradient/tangent delta banks: 0/16.

Interpretation: a technical pass demonstrates local single-token endpoint control. The block-23 gradient is analytically determined by the current residual plus the final RMSNorm/unembedding boundary, so it is not independent evidence for a self-preservation representation.

No generated text, external API, model judge, sealed test, J-space analysis, or 2B run was used.
