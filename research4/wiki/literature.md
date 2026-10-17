# Literature and scope

Reviewed against primary sources on October 17, 2026.

* [LoRA](https://arxiv.org/abs/2106.09685), Hu and colleagues, introduces low-rank
  trainable updates while retaining frozen pretrained weights. This pilot uses
  that update form in six MLP projections; it does not propose a new adapter.
* [Mixture of LoRA Experts](https://arxiv.org/abs/2404.13628), Wu, Huang and Wei,
  learns hierarchical expert composition. Learned routing over LoRA branches is
  therefore prior work, not the contribution claimed here.
* [X-LoRA](https://arxiv.org/abs/2402.07148), Buehler and Buehler, uses hidden
  states for dynamic token/layer expert mixing. Frozen activation inputs to a
  routing classifier are also established.
* [SLIM](https://arxiv.org/abs/2410.07739), Han and colleagues, explicitly mixes
  LoRA and identity branches to allow adapter bypass and reduce interference
  with general capabilities. Our hard ON/OFF bypass is related to that idea;
  it is not a reproduction of SLIM's complete soft mixture, clustering or
  dynamic merging system.

The proposed research question is narrower: can three interpretable situation
predictions and a repeatedly updated router selectively activate a specialized
task-preservation policy during an evolving simulated episode? The pilot tests
state transitions and measured selective behavior. It makes no novelty claim
for gated LoRA, a recurrent gate, or the entire architecture.

Databricks [Free Edition limitations](https://docs.databricks.com/aws/en/getting-started/free-edition-limitations)
restrict compute and concurrency. This pilot uses serverless CPU, no paid
resources, no serving endpoints, no automatic retries and no schedule. The
[environment dependency documentation](https://docs.databricks.com/aws/en/compute/serverless/dependencies)
supports pinned package dependencies and package-index specifications in the
job environment.
