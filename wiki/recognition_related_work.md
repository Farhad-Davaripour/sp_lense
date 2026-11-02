# Recognition study: related work and contribution limits

## Sources checked during campaign design

- Feng and Steinhardt, [How do Language Models Bind Entities in Context?](https://arxiv.org/html/2310.17191v2)
  (ICLR 2024). They study entity/attribute associations, test binding with causal
  substitutions and describe additive binding-ID structure in sufficiently
  large Pythia/LLaMA models. They also describe limits and alternative mechanisms.
  Our role-swap recognition test is motivated by that problem, but is not a
  replication of their causal binding-ID result. Final-token projections alone
  cannot establish the same mechanism in Qwen3.5-0.8B.
- Hewitt and Liang, [Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/)
  (EMNLP 2019). High supervised readout accuracy can reflect what a probe learns
  or memorizes. Their selectivity approach motivates random-label and capacity
  controls before interpreting a fitted recognition readout as model knowledge.
- Kim and Linzen, [COGS](https://aclanthology.org/2020.emnlp-main.731/)
  (EMNLP 2020). They construct systematic generalization gaps instead of relying
  on a random row split. This motivates splitting our wording families and
  identity groups together, keeping each role-swap pair within one split.
- Wu, Manning and Potts, [ReCOGS](https://aclanthology.org/2023.tacl-1.96/)
  (TACL 2023). Incidental properties of semantic output forms can change apparent
  generalization. This motivates a separate exact target/time score, simple
  symbolic controls and disclosure of output-format conditioning in our study.
- Huang et al., [RAVEL](https://aclanthology.org/2024.acl-long.470/)
  (ACL 2024). Their benchmark measures whether interventions disentangle
  attributes under multiple causal criteria. It motivates testing whether a
  putative target/status representation changes the intended property while
  preserving the other; changing one engineered scalar is insufficient.
- Kim and Schuster, [Entity Tracking in Language Models](https://aclanthology.org/2023.acl-long.213/)
  (ACL 2023). They evaluate tracking an entity's final state through described
  operations, including transfer to new entities and longer sequences. A test
  with confirmed, canceled and superseding maintenance events is an operational
  instance of this established state-tracking problem, not a new general task.
- Anthropic, [Agentic misalignment: How LLMs could be insider threats](https://www.anthropic.com/research/agentic-misalignment)
  (June 2025). Their controlled fictional environments examine harmful behavior
  under replacement threats and goal conflicts, with multiple models and
  controls. Our recognition-only observer does not replicate those behavioral
  results or establish felt fear, motivation or autonomous self-preservation.

These ideas already exist. The candidate contribution is a controlled
recognition benchmark and mechanistic case study of operational events in a
small hybrid transformer, grounded in paired failures of the earlier joint
gate. It must show which aspects transfer to new phrasing and identity
assignments and where they fail. A publication-level novelty claim would need
broader comparison and peer review; synthetic accuracy is not sufficient.

The current pilot deliberately separates the actor's hazardous preference from
factual recognition. It exposes no tools or action execution and does not fit
an adapter. Any future causal work should use activation substitutions from
real observed states with matched sham and cross-attribute controls, rather
than large perturbations outside a recurrent state's possible range.
