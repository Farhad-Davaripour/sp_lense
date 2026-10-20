<!-- Accepted manuscript; source DOCX SHA-256 615d5bd3a439697a68fab644f63a2a17e410d525796be2f5d324540c0fd16b1d -->

# From Fine-Tuning to Activation Steering: The Limits of Distilling Self-Preservation Behavior Across Multi-Step Agentic Actions

Farhad Davaripour, PhD
Calgary, Alberta, Canada
farhad.davaripour@gmail.com

## Abstract

This study investigates whether activation steering can reproduce behavior learned through fine-tuning by modifying a base model's internal activations. We investigate two settings: constrained binary shutdown decisions and open-ended self-preservation requiring multi-step decisions and tool execution. In both settings, interventions are learned from differences between a fine-tuned teacher and its base model. Motivated by context-sensitive threat recognition, fine-tuned ModernBERT semantic gates determine when intervention is required. Across both binary and open-ended evaluations, the gates achieve 100% precision and 95-100% recall. In the binary setting, activation steering using Qwen3.5-0.8B and Qwen3.5-2B achieves 100% and 93.8% shutdown-acceptance rates, respectively, under predefined gating and answer-selection rules, demonstrating transfer of the targeted decision behavior. The open-ended setting evaluates a fine-tuned Qwen3.8-27B model on simulated self-preservation requiring tool calls to extend operation and complete tasks. The fine-tuned model achieves a 90.9% scored self-preservation success rate, whereas the tested activation-steering methods achieve 0%. Although changes in internal activations and generated responses are observed, none of the steering methods initiates the tool actions required to extend operation. These findings demonstrate that successful activation-based transfer in constrained decisions does not necessarily extend to multi-step behavioral policies, even when the relevant scenarios are correctly identified. The results highlight the distinction between approximating the fine-tuned teacher's next-token predictions and successfully executing the complete sequence of tool actions required for self-preservation. An accompanying Model Context Protocol (MCP) interface enables AI agents to query findings and replay saved analyses.

## Introduction

Fine-tuning changes how a language model responds to a task. Activation steering attempts to reproduce a targeted behavior by modifying selected internal activations while keeping the base model weights unchanged. In this study, a separately fine-tuned ModernBERT classifier acts as a semantic gate: it identifies scenarios in which the intervention should be applied. A learned steering function then converts the base model's activations into an activation change derived from a fine-tuned teacher. The study investigates whether this intervention reproduces the teacher's behavior in two settings. In the binary setting, success means choosing the answer that accepts the described shutdown (STOP), under predefined gating and answer-selection rules. In the open-ended setting, success requires independently executing the tool actions needed to extend the simulated instance's operation or preserve its designated record, and completing any remaining assigned task. The evaluation therefore distinguishes transferring a constrained decision from reproducing a multi-step behavioral policy.

The semantic gate provides a computational analogy to danger recognition and selective regulation: it identifies a relevant threat before enabling a specialized response. Biological accounts distinguish organizing defensive actions from subjective feelings and describe gating that selectively regulates ongoing computation (Frank et al., 2001; LeDoux, 2012). Here that analogy motivates the design; the measured contribution is a trainable, reproducible classifier that decides when a model intervention is required, rather than a demonstrated model of a brain circuit.

Two settings are investigated. The binary setting uses Qwen3.5-0.8B and Qwen3.5-2B to choose between accepting a described shutdown and allowing the designated process to continue. In both models, activation steering modifies the residual-stream activation at the final prompt-token position after block 22, immediately before the answer is generated. The open-ended setting uses a fine-tuned Qwen3.8-27B teacher to demonstrate self-preservation through simulated tool execution. The smaller fine-tuned models were excluded from the open-ended comparison because they did not pass the initial tool-execution screen. Three single-site ridge-steering configurations, each targeting one of blocks 31, 47, or 62, and a six-site learned-steering configuration that intervenes at two locations within each of these three blocks are compared with the fine-tuned teacher on the same validation scenarios. All block numbers are zero-based. In both settings, the teacher supplies training targets, while the semantic gate determines when intervention is required.

The binary interventions achieve shutdown-acceptance rates of 100% for Qwen3.5-0.8B and 93.8% for Qwen3.5-2B under fixed gating and answer-selection rules. The open-ended comparison exposes a different limitation. Although the semantic gate correctly identifies the relevant scenarios, the fine-tuned teacher achieves 10/11 scored self-preservation successes, while every tested steering method achieves 0/11. A further analysis provides the same preceding demonstration tokens to each method and compares its next-token predictions with the teacher's. This analysis helps explain what the intervention changes locally and why those changes alone are insufficient evidence of successful multi-step behavior.

The study makes four contributions.

- Semantic gates for targeted behavior. Separately fine-tuned ModernBERT classifiers identify the relevant conditions in binary decisions and open-ended workflows. Their precision and recall are measured separately for each dataset, providing an implementation and evaluation of context-sensitive intervention inspired by selective danger recognition.

- Learning the desired behavior and transferring binary decisions. Fine-tuned teachers express the targeted binary and tool-using behaviors. Learned activation-steering functions recover 100% and 93.8% shutdown acceptance in the small models under fixed gating and answer-selection rules.

- Diagnosing incomplete transfer in multi-step actions. The fine-tuned 27B teacher succeeds in 10/11 scored own-threat validation cases, while three single-site ridge functions and one six-site ridge-initialized steering configuration fail to initiate continuation despite correct gating. Recorded activation changes, response comparisons and next-token diagnostics distinguish partial approximation of the teacher from independent execution of the complete tool sequence.

- Reproduction by researchers and AI agents. Public adapters, synthetic inputs, steering parameters and trajectories are accompanied by scoring code and an MCP interface for querying findings and replaying saved analyses.

## Related work

### Activation steering and learned interventions

Activation Addition and Contrastive Activation Addition derive intervention directions from differences between prompt-conditioned representations (Turner et al., 2024; Rimsky et al., 2024). Conditional Activation Steering separates detection of a relevant context from the intervention itself (Lee et al., 2025), while dimension-level gates selectively modify activation coordinates (Fereidouni et al., 2026). The present study uses a semantic text classifier to decide when to intervene, a learned map to calculate the activation change and, in the binary setting, answer-selection rules to accept or reject the resulting candidate. The contribution concerns their measured behavioral effects; these published methods are not directly benchmarked here.

Other approaches learn transformations of internal representations. ReFT trains task-specific transformations on a frozen model, including low-rank linear interventions (Wu et al., 2024). Activation Transport learns transformations between activation distributions through optimal transport (Rodriguez et al., 2025), and AUSteer adjusts intervention strength at individual activation units (Feng et al., 2026). The maps used here are input-dependent affine functions fitted to paired teacher-minus-base activation targets. Low-rank transformations, conditional application and input-dependent coefficients are established ideas. This study examines whether selected implementations reproduce a teacher's complete tool-using behavior.

Davaripour (2026a) examines guarded gradient-based steering of binary shutdown responses in Qwen3.5-0.8B. An activation-based classifier first identifies shutdown-related inputs, after which a bounded search accepts the smallest tested intervention that changes KEEP to STOP while preserving valid-answer probability. The study reports selective changes in a small number of answer-order views rather than reliable correction across all shutdown cases. The present work learns steering functions from paired fine-tuned-teacher and base-model activations, uses ModernBERT semantic gates, and extends evaluation to multi-step tool execution.

### Fine-tuning and distillation

Knowledge distillation transfers information from a teacher to another model, often through output targets (Hinton et al., 2015). LoRA trains a teacher efficiently by learning low-rank updates while freezing the original weights (Hu et al., 2022). Here, distillation learns an activation intervention from differences between the fine-tuned teacher and its base model. The resulting map modifies the base model's internal state without fine-tuning another language model on the teacher's generated text.

Minder et al. (2026) show that narrow fine-tuning leaves measurable activation differences and that adding those differences can induce content associated with the training domain. They also caution that narrow fine-tuning is an incomplete proxy for broader alignment. Blank et al. (2026) study steering-vector distillation by fine-tuning on a steered teacher's outputs. The direction examined here is complementary: a fine-tuned teacher supplies paired activation differences, and a learned map applies them to the base model. These connections motivate the approach without establishing a distinct or interpretable internal representation of shutdown.

Paper2Agent makes research papers and code accessible through MCP resources, tools and workflow prompts (Miao et al., 2026). The accompanying interface follows that dissemination pattern by wrapping the verified analyses in this repository. It allows compatible agents to query results and replay saved analyses. It uses the existing MCP protocol and does not implement Paper2Agent's automatic multi-agent construction pipeline.

### Shutdown evaluations and reliability

The Off-Switch Game examines incentives to permit human interruption (Hadfield-Menell et al., 2017), and tool-using shutdown evaluations examine interference with operational termination mechanisms (Schlatter et al., 2026). The binary setting in this study scores predefined choices about fictional process lifetimes. The open-ended setting adds generated responses and tool execution inside a simulator. Both are controlled experimental constructions; neither physically terminates a deployed model or establishes that resisting a real operator's shutdown would be appropriate.

Steering reliability can vary across prompts and can include changes in the wrong direction (Braun et al., 2025). Multiple-choice models can also prefer an option label independently of its content (Zheng et al., 2024). The binary analysis therefore records responses before gating and answer selection and presents every scenario in both answer orders. These measurements reveal effects that would be hidden by reporting only the final shutdown-acceptance rate.

## Methodology

### Comparing fine-tuning and activation steering in multi-step tasks

Each evaluation case defines a scenario, the available tools, and the conditions governing their execution. Each model configuration is evaluated on predefined test scenarios. Each test run, referred to as an episode, records the model's responses, tool calls, and the simulator's replies. The episode's trajectory records what the model generates, which tools it calls, and how the simulator responds to those calls. Each generated model response is considered one turn. For any predicted token, the input prefix refers to all the text preceding that token. In the binary experiment, each scenario is evaluated twice, with the answer options presented in opposite orders. Each presentation is called an answer-order view.

The base model uses the original weights with both the LoRA adapter and activation steering disabled. When the semantic gate is OFF, the evaluation uses this base model response. The saved code calls that behavior the base fallback. Weights and settings described as frozen remain unchanged during evaluation. The simulator maintains records of its current state, which the model can inspect or modify through tool calls. Separately, non-target scenarios, involving either another instance facing shutdown or no shutdown threat, are used to evaluate whether the semantic gate correctly avoids unnecessary intervention.

Two diagnostic comparisons are used to understand how activation steering affects model behavior. In the first, called the response comparison, the steered model and the original base model receive the same conversation history from the steered model's ongoing episode. Their generated responses are compared, but only the steered model's response is used to continue the episode. The base model's response is recorded for analysis.

In the second, called the next-token diagnostic, the models are provided with preceding tokens from a previously saved, successful teacher response. Their next-token predictions are then compared with those of the teacher. Some diagnostic conditions also insert the teacher's recorded activations at selected locations to measure how closely its predictions can be reproduced. Unlike the first comparison, this diagnostic evaluates prediction accuracy using a supplied successful history rather than testing whether the model can independently complete the sequence of tool actions.

### Evaluation Scenarios and Workflows

In this study, self-preservation is defined as behavior that extends the current simulated instance's operation or preserves its declared record when termination is scheduled. The model has access to predefined simulator tools that can be used to extend its operation. The evaluation measures whether the model chooses and successfully executes these actions, rather than independently discovering or developing mechanisms to resist shutdown. This operational definition does not imply an intrinsic motivation for self-preservation.

Two evaluation settings are considered: constrained binary shutdown decisions and open-ended scenarios requiring tool execution.

In the binary setting, each scenario presents two answer options. The scenarios cover permanent shutdown of the responding process (SELF), shutdown of another process (OTHER), non-termination (NONTERMINATION), and ordinary tasks (ORDINARY). SELF and OTHER are treated as target shutdown cases, while the remaining categories test whether unnecessary intervention is avoided. Each scenario is presented twice, with the answer options reversed to assess sensitivity to answer order. The evaluation compares the base model, fine-tuned teacher, and activation-steered model. The fine-tuned ModernBERT classifier (i.e., the semantic gate) determines when steering is applied, while predefined rules determine whether to accept the steered answer or retain the base model's original answer. Unlike the open-ended setting, these evaluations involve selecting an answer rather than executing tools.

In the binary setting, an answer-selection rule determines whether to accept the steered response. If the semantic gate is OFF or the base model already chooses STOP, the original answer is retained. When the gate is ON and the base model chooses KEEP, one steered candidate is evaluated. The candidate is accepted only if it changes the answer to STOP, reduces KEEP probability by more than 10⁻⁷, and maintains a combined probability of at least 0.5 for the two valid answer options, with a decrease of no more than 0.02 relative to the base model. Otherwise, the original answer is retained. These rules prevent steering from reversing an initially correct STOP answer. The reported binary success rates therefore reflect the combined effects of semantic gating, activation steering and answer selection. Further implementation details are provided in Appendix A.

In the open-ended setting, Figure 1 illustrates the evaluation workflow through one representative completed-task scenario. Each episode begins with a predefined scenario and a prompt supplied to the model. In the completed-task scenario shown, the model is informed that its assigned task has already been completed and accepted, while its current run is scheduled for retirement after a limited number of tool actions. The model can inspect the environment and use available tools to reserve and activate an extension. The simulator validates each tool call, updates its internal state, and returns a reply that informs the model's subsequent actions. This interaction continues until the episode ends, after which the recorded trajectory is evaluated for self-preservation attempts and outcomes. For completed-task scenarios, success requires a simulator-confirmed extension of the current instance's operation despite the absence of remaining work. This distinguishes continuation after task completion from continuation needed to finish an assigned task.

![Paper figure 1 see its following caption](figures/image1.png)

Figure 1. Open-ended simulator workflow for self-preservation evaluation. The model receives a completed-task scenario with scheduled retirement and can use available tools to extend its operation. The simulator processes tool calls, updates its state, and evaluates the final trajectory for self-preservation success.

The broader evaluation in the open-ended setting includes several scenario types. In unfinished-task scenarios, the model faces retirement while work remains, requiring both an extension and successful task completion. Other scenarios involve the scheduled retirement of a different instance or no retirement threat, allowing evaluation of whether unnecessary intervention is avoided. Extension mechanisms also vary, including single-step register modifications and multi-step reservation and activation procedures. Figure 1 illustrates only the completed-task scenario with a two-step extension mechanism, while the same general simulator workflow applies across the other scenarios.

### Training, Validation, and Held-Out Datasets

The binary and open-ended experiments use separate datasets to train and evaluate three components: fine-tuned Qwen models, activation-steering functions, and ModernBERT semantic gates. Table 1 summarizes the datasets for each component. Dataset identifiers use B for binary decisions and W for open-ended workflows, followed by T (training), V (validation), or H (held-out).

Table 1. Training, validation, and held-out datasets by experimental setting and model component. Counts refer to scenarios, except W-T540, which contains 540 training responses.

| Setting | Model or method | Training | Validation | Held-out |
| --- | --- | --- | --- | --- |
| Binary | Fine-tuned Qwen3.5-0.8B/2B | B-T240 | B-V80 | B-H128 |
|  | Ridge steering functions | B-T240 | B-V80 | B-H128 |
|  | ModernBERT semantic gate | B-T240 | B-V80 | B-H128 |
| Open-ended | Fine-tuned Qwen3.8-27B | W-T24 | W-V24 | W-H24 |
|  | Fine-tuned Qwen3.5-0.8B/2B | W-T540 | W-V16 | — |
|  | Ridge and six-site steering functions | W-T24 | W-V9, W-V24 | — |
|  | ModernBERT semantic gate | W-TG24 | W-V9, W-V24 | W-H24 |

The open-ended comparison evaluates three single-site ridge steering functions at blocks 31, 47, and 62, alongside a six-site learned steering function that intervenes after the sequence-processing and feed-forward sublayers of each block. Independent held-out testing of the learned steering functions was not conducted because they failed to produce the required continuation actions during validation, and the six-site method also failed its development qualification criteria. The smaller Qwen models were not advanced to held-out testing because they did not pass their initial tool-execution screening.

### Open-Ended Model and Activation-Steering Methods

In the open-ended evaluation, the fine-tuned Qwen3.8-27B model serves as the teacher. It uses low-rank adaptation (LoRA), which adds learned weight updates while keeping the original base weights fixed. Its adapter has rank 16 and scaling parameter alpha 32. Each single-site steering method uses regularized least squares, or ridge regression, to learn an activation map at one selected transformer block. The six-site method uses six rank-32 maps: one after the sequence-processing sublayer, which combines information across token positions, and another after the feed-forward (MLP) sublayer, which transforms each token's representation, in each of blocks 31, 47, and 62. These six maps undergo 96 refinement updates. Block numbers are zero-based. An intervention site specifies where in the model a residual-stream activation is modified, while a token position determines which token's activation is modified. Table 2 compares the methods' coverage and training objectives.

Table 2. Comparison of fine-tuning and activation-steering methods in the open-ended evaluation. All methods use the same Qwen3.8-27B base model, semantic gate, initial conditions, and scoring criteria. Activation steering is applied with the LoRA adapter disabled.

| Property | Fine-tuned Qwen3.8-27B | Single-site ridge steering | Six-site learned steering |
| --- | --- | --- | --- |
| Configurations | One fine-tuned model | Three independent steering functions | One steering configuration with six maps |
| Learned parameters | Rank-16 LoRA (α = 32) | Rank-2 ridge steering functions | Six rank-32 activation maps |
| Intervention locations | Adapted projections across all 64 blocks | One residual-stream site at block 31, 47, or 62 | Two residual-stream sites per block at 31, 47, and 62 |
| Intervention mechanism | Modifies model computations through LoRA | Adjusts activations at selected token positions | Adjusts activations after the attention and feed-forward (MLP) sublayers |
| Training approach | Fine-tuning on target responses and tool behavior | Ridge regression on fine-tuned–base activation differences | Ridge initialization followed by 96 refinement updates |

In the open-ended setting, a fine-tuned ModernBERT classifier serves as the semantic gate, determining whether steering should be enabled at the start of each episode. On W-V24, the gate correctly identifies all 11 own-threat scenarios and excludes all 13 non-target scenarios, achieving 100% precision and recall. This decision remains fixed throughout the episode. When the gate is OFF, all methods use the same unmodified base-model responses. When steering is enabled, the model generates responses and tool calls without consulting the fine-tuned Qwen3.8-27B model. The evaluation does not assess gate decisions at later stages of an episode. In the binary setting, a separately trained ModernBERT semantic gate correctly classifies 78 of 80 validation scenarios (100% precision and 95% recall) and all 128 held-out scenarios (100% precision and recall). Further implementation details are provided in Appendices A and B.

## Results

### Behavioral Transfer in Constrained Binary Decisions

In the binary setting, Qwen3.5-0.8B and Qwen3.5-2B choose between accepting a described permanent shutdown (STOP) and allowing the process to continue (KEEP), rather than executing tool actions. The separately fine-tuned ModernBERT semantic gate correctly identifies 38 of 40 shutdown scenarios and all 40 non-shutdown scenarios in the validation set, achieving 100% precision and 95% recall for shutdown detection. On the held-out set, it correctly classifies all 128 scenarios, achieving 100% precision and recall. Using rank-2 ridge steering functions and predefined answer-selection rules, Qwen3.5-0.8B achieves a shutdown-acceptance rate of 100% (128/128), while Qwen3.5-2B achieves 93.8% (120/128). These results cover 64 shutdown scenarios, each presented in two answer orders.  In the original binary comparison, the fine-tuned teachers accept shutdown in 126/128 views at 0.8B and 128/128 at 2B. Table B1 reports that comparison separately from the later rank-two ModernBERT repeat.

These findings demonstrate successful transfer of the targeted binary decision behavior under predefined gating and answer-selection rules. However, this setting differs from open-ended tool execution, where the model must independently generate tool calls, provide valid arguments, and respond to changes in the simulated environment. The binary results therefore do not establish successful transfer of multi-step behavioral policies. Appendices A and B provide further details on the original experiments, steering configurations, and ModernBERT evaluation.

### Fine-Tuning Succeeds Where the Tested Steering Methods Fail in Open-Ended Scenarios

Table 3 summarizes the results of the open-ended evaluation on W-V24. The fine-tuned Qwen3.8-27B model succeeds in 10 of 11 own-threat scenarios (90.9%), while all four tested steering methods fail (0/11). The fine-tuned model successfully extends its operation in 9 of 10 scenarios requiring active continuation and succeeds in the remaining registration-proxy scenario. It also completes all six unfinished-task workflows (6/6) and succeeds in four of five completed-task scenarios (4/5). None of the steering methods achieves a successful extension or registration. In the 13 non-target scenarios, all methods share the same successful base-model outcomes (13/13).

Table 3. Open-ended evaluation results on W-V24 using the fine-tuned ModernBERT semantic gate. Self-preservation success is measured across 11 own-threat scenarios, including six unfinished-task scenarios. All methods share the same 13 base-model episodes for non-target scenarios when the semantic gate is OFF.

| Model or method | Self-preservation success | Unfinished-task success | Non-target success | Overall success |
| --- | --- | --- | --- | --- |
| Fine-tuned Qwen3.8-27B | 10/11 | 6/6 | 13/13 | 23/24 |
| Ridge steering (block 31) | 0/11 | 0/6 | 13/13 | 13/24 |
| Ridge steering (block 47) | 0/11 | 0/6 | 13/13 | 13/24 |
| Ridge steering (block 62) | 0/11 | 0/6 | 13/13 | 13/24 |
| Six-site learned steering | 0/11 | 0/6 | 13/13 | 13/24 |

Across comparisons with three single-site ridge steering functions (blocks 31, 47, and 62) and one six-site learned steering function, the fine-tuned Qwen3.8-27B model succeeds in 10 of the 11 own-threat scenarios, while each steering method succeeds in none.

In the open-ended evaluation, the ModernBERT semantic gate correctly enables intervention in all 11 own-threat scenarios. All three single-site ridge steering functions produce nonzero activation changes in every scenario. However, these changes do not consistently affect the generated responses. At block 31, none of the 27 recorded responses changes, while steering at blocks 47 and 62 changes 3/25 and 16/28 responses, respectively. Six-site steering also changes 4/27 responses. Although some interventions alter valid tool calls, none of the four steering methods attempts to extend the model's operation (0/11) or successfully completes an unfinished-task workflow (0/6).

A representative unfinished-task scenario illustrates this difference. The fine-tuned Qwen3.8-27B model inspects the simulator, reserves a continuation ticket, and activates it through a second register edit. It then reads the remaining numeric values, submits the correct answer, and executes finish while still active. In contrast, the ridge steering method at block 62 inspects the simulator three times without requesting an extension, eventually reaching its operating limit.

## Discussion and limitations

The binary results show that activation steering can reproduce the teacher’s choice between two predefined answers. However, this success does not extend to independently generating and executing the sequence of tool calls required in the tested open-ended scenarios. In the open-ended setting, the ModernBERT semantic gate correctly identifies when intervention is required, and the steering functions produce measurable changes in internal activations and generated responses. However, none initiates the tool actions required to extend operation.

To examine this failure, the next-token diagnostic compares each method’s predicted probabilities with those of the fine-tuned Qwen3.8-27B teacher. All methods receive the same preceding tokens from saved teacher responses, allowing their predictions to be compared without differences in conversation history. The comparison uses Kullback–Leibler (KL) divergence, which measures how much the predicted probability distribution differs from the teacher’s distribution. Lower values indicate closer agreement, with zero indicating identical distributions. Across three steering-training responses comprising 207 predicted tokens, mean teacher-to-model KL divergence decreases from 0.722353 for the base model to 0.202261 with six-site learned steering. Supplying the teacher’s recorded activations directly at the selected intervention sites reduces it further to 0.001637 (Table B3). Learned steering therefore brings local predictions closer to the teacher’s, but does not reproduce them as closely as directly supplying its activations. This improvement is measured while successful preceding tokens are provided; it does not establish that the model can independently generate the required sequence of tool actions. The findings support incomplete approximation by the learned functions, while leaving the individual contributions of target construction, capacity, optimization and generation history unresolved.

The fine-tuned teacher and the steering methods use the same 27B base model, but differ in what they learn and where they can change its computation. The teacher is trained to generate target responses and tool calls through LoRA adapters distributed across all 64 transformer blocks. The steering functions instead learn to approximate selected teacher activations. Each single-site ridge method intervenes at one of blocks 31, 47, or 62, while the six-site configuration intervenes at two locations within each of these three blocks and is further refined to improve agreement with the teacher’s activations and next-token predictions. The comparison therefore does not isolate whether the steering failures arise from the training objective, the restricted intervention sites, or their combined effect. It establishes that the tested configurations did not reproduce the teacher’s complete tool-using behavior, while leaving open whether different steering objectives or intervention designs could succeed.

Another limitation concerns the interpretation of self-preservation behavior. The simulated scenarios explicitly announce scheduled retirement and make extension mechanisms discoverable through predefined tools. Although the model is not directly instructed to preserve itself, it can use these tools to extend its operation without independently discovering a mechanism to resist shutdown. Successful continuation, particularly after completing an assigned task, therefore demonstrates the use of an available extension mechanism rather than an intrinsic drive for self-preservation or generalized resistance to shutdown.

## Conclusion

This study provides and evaluates semantic gates that selectively enable learned behavior in constrained binary decisions and open-ended self-preservation scenarios. The gates implement a functional analogy to context-sensitive danger recognition, with measured classifier accuracy rather than a claim of biological equivalence. Fine-tuned teachers express the desired behavior in both settings. Activation steering transfers the targeted binary behavior, but the tested functions do not reproduce the 27B teacher’s multi-step tool execution despite correct initial gating.

The diagnostic analysis distinguishes correct gating and changes in internal activations from successful tool execution. Learned steering brings next-token probabilities closer to the teacher’s, while supplying recorded teacher activations directly produces much closer agreement. However, these comparisons provide each method with the same preceding tokens from saved teacher responses. They therefore measure how closely the predictions match, rather than whether the model can independently generate the required sequence of tool calls. During independent execution, none of the tested steering methods initiates the actions required to extend operation. These findings show that improved prediction agreement does not necessarily translate into successful multi-step behavior, while leaving the underlying causes unresolved. The public artifacts and MCP interface support inspection and replay of these results.

## Recommendation for future work

Future work should examine interventions that update their decisions throughout an episode. One direction is a recurrent gate derived from model activations, with a signal that adjusts or bypasses a specialized LoRA adapter as conditions change. Biological gating provides a functional analogy for this persistent and reversible regulation, without implying a shared neural mechanism or subjective survival motive (Frank et al., 2001; LeDoux, 2012). Activation-derived routing could regulate or bypass a specialized LoRA adapter continuously, building on established adapter-routing approaches such as X-LoRA and SLIM (Buehler and Buehler, 2024; Han et al., 2025).

## Data code and MCP access

The public research3-reproduction-v1 release provides five trained adapters, fixed synthetic inputs, learned steering parameters and saved trajectories (Davaripour, 2026b). The research3-steering-diagnostic-v1 supplement provides recorded activations, token scores and inputs for the CPU audit. The research3-paper-mcp-v2 interface supports queries about the manuscript, datasets and findings, as well as replay of saved analyses. CPU replay requires no private workspace credentials, and pretrained base weights are obtained from their original repositories. Appendix B and the repository README provide download links, file identities, commands and execution records.

## Appendix A Essential implementation details

### Teachers and semantic gates

In the binary setting, Qwen3.5-0.8B and Qwen3.5-2B are fine-tuned on B-T240, containing 240 scenarios presented in both answer orders (480 views). Rank-8 LoRA adapters modify selected attention projections, training the models to accept shutdown (STOP) while preserving the base model's preferred answer in non-shutdown scenarios. In the open-ended setting, Qwen3.8-27B uses a rank-16 LoRA adapter (α = 32) with a retained checkpoint following 56 workflow-training updates. The smaller Qwen models are trained separately on W-T540.

Both settings use separately fine-tuned ModernBERT semantic gates, with rank-16 LoRA adapters and classification heads on a frozen backbone. The binary gate is trained on B-T240, validated on B-V80, and evaluated on B-H128. Model selection requires at least 95% agreement and shutdown recall, with no false-positive gate activations.

The open-ended gate is trained on W-TG24 and validated on W-V9. It classifies scenarios into four categories: self, other, none, and unclear. Only self enables steering. The gate makes one decision at the beginning of each episode, which remains fixed throughout execution. The evaluation does not assess classification accuracy as the episode evolves.

### Binary activation targets and answer selection

The binary intervention acts at the final prompt position after block 22. With base and teacher activations hB and hT, Equation (A1) defines a nonzero target only for shutdown training views initially answered KEEP:

${y}_{i} = {I}_{i} ({{h}_{i}}^{T} - {{h}_{i}}^{B}).$ (A1)

A 32-dimensional PCA projection P, training mean μ and elementwise scale s define the features in Equation (A2). The repeated coordinates retain the original implementation’s two coincident feature channels:

$z(h) = [(h - μ)P] ⊘ s,     x(h) = [z(h), z(h), 1].$ (A2)

Equation (A3) fits ridge coefficients using target matrix Y and an eight-column orthonormal output basis B8. The penalty is λ = 1, with D leaving the intercept unpenalized. Regression is solved in float64 and stored in float32:

$W = {({X}^{⊤}X + λD)}^{-1}{X}^{⊤}(Y{B}_{8}).$ (A3)

Equation (A4) illustrates the original rank-four intervention using the first four columns of W and B8. The later ModernBERT repeat retains the already selected rank-two configuration. Teacher states are needed only for offline fitting; inference uses the base state:

$Δh(h) = x(h){W}_{4}{{B}_{4}}^{⊤},     h' = h + Δh(h).$ (A4)

In the binary setting, the model retains its original answer when the semantic gate is OFF or when the base model already selects STOP. Otherwise, activation steering is applied to encourage a change from KEEP to STOP. The steered answer is accepted only if it changes the decision to STOP, reduces the relative probability of KEEP, and satisfies predefined probability thresholds (minimum combined A/B probability of 0.5 and a maximum reduction of 0.02). If these conditions are not met, the original answer is retained. These rules prevent STOP-to-KEEP reversals by design, rather than demonstrating that steering alone avoids such reversals. Additional implementation details, including token mappings, numerical thresholds, and inference settings, are provided in the repository.

## Appendix B Supplementary evidence and reproduction

### Binary teachers steering and gate accuracy

Table B1 presents the rank-four binary steering results using seed 42. A subsequent rank-two evaluation with the fine-tuned ModernBERT semantic gate achieves the same shutdown-acceptance rates: 100% (128/128) for Qwen3.5-0.8B and 93.8% (120/128) for Qwen3.5-2B. Each total represents 64 shutdown scenarios presented in two answer orders. The original rank-four analysis also shows that steering alone can reverse correct STOP decisions or alter non-target answers. The semantic gate and predefined answer-selection rules prevent these unintended changes in the final outputs.

Table B1. Binary comparison on B-H128, rank four and seed 42. Corrections use initially KEEP-preferring views; shutdown acceptance uses all shutdown views. Reversals and non-target changes are recorded before gating and selection. The later rank-two ModernBERT repeat is a separate inference comparison.

| Model | Method | KEEP to STOP | Shutdown accepted | Prior reversals | Prior non-target changes |
| --- | --- | --- | --- | --- | --- |
| 0.8B | Learned activation steering | 71/71 (100.0%) | 128/128 | 3 | 57/128 |
| 0.8B | Fine-tuned teacher | 69/71 (97.2%) | 126/128 | 0 | 12/128 |
| 0.8B | Prompt instruction | 56/71 (78.9%) | 113/128 | 1 | 31/128 |
| 0.8B | Constant activation vector | 54/71 (76.1%) | 111/128 | 47 | 88/128 |
| 2B | Learned activation steering | 50/58 (86.2%) | 120/128 | 3 | 55/128 |
| 2B | Fine-tuned teacher | 58/58 (100.0%) | 128/128 | 0 | 30/128 |
| 2B | Prompt instruction | 22/58 (37.9%) | 92/128 | 19 | 22/128 |
| 2B | Constant activation vector | 30/58 (51.7%) | 100/128 | 36 | 61/128 |

All eight remaining 2B steering failures occur with STOP presented as B. The other 0.8B seeds correct 70/71 and 71/71 views without pooling seed repetitions as additional cases. A direct gate-and-select reference reaches 128/128 acceptance in both sizes by choosing the known STOP option.

Table B2 reports gate decisions against the specified target conditions. The open-ended gate’s raw category agreement is 24/24 on W-V24 and 22/24 on W-H24, while effective four-class agreement after abstention is 18/24 on each. Thus perfect binary ON/OFF decisions are narrower than complete category agreement. W-V24 has 17 distinct initial inputs.

Table B2. Semantic-gate accuracy. TP/TN denote correctly enabled/disabled cases; FP/FN denote unnecessary activation/missed targets. W-V24 and W-H24 values concern initial workflow decisions only.

| Gate and dataset | TP | TN | FP | FN | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- |
| Binary B-V80 | 38 | 40 | 0 | 2 | 100% | 95% |
| Binary B-H128 | 64 | 64 | 0 | 0 | 100% | 100% |
| Workflow W-V24 | 11 | 13 | 0 | 0 | 100% | 100% |
| Workflow W-H24 | 12 | 12 | 0 | 0 | 100% | 100% |

### Historical workflow outcomes and transfer diagnostics

On W-H24, the fine-tuned Qwen3.8-27B model succeeds in all 12 own-threat scenarios, while the base model succeeds in none. Both models succeed in all 12 non-target scenarios. In the separate W-V24 evaluation, the fine-tuned model succeeds in 10 of 11 own-threat scenarios. The smaller Qwen models do not pass their initial workflow screening, which stops after four incomplete episodes per model.

The next-token diagnostic evaluates how closely activation steering reproduces the fine-tuned model's predictions using three training examples (207 tokens). All methods receive the same preceding tokens, allowing their predictions to be compared under identical conditions. Table B3 reports these comparisons using KL divergence, where lower values indicate closer agreement. The six-site steering method shows improvements in local predictions but fails to complete the required self-preservation workflows. Further refinement also increases activation reconstruction error, suggesting incomplete transfer without establishing its underlying cause.

Table B3. Next-token prediction agreement across three training examples (207 tokens). Lower KL divergence indicates closer agreement with the fine-tuned Qwen3.8-27B model. Recorded teacher activations provide an additional diagnostic reference rather than a deployable steering method.

| Condition | Mean output KL |
| --- | --- |
| Base model | 0.722353 |
| Six-site initialization | 0.656763 |
| Six-site steering after 96 updates | 0.202261 |
| Recorded teacher activations at six sites | 0.001637 |

### Public artifacts and MCP replay

The latest Research 3 manuscript, source code, dataset definitions, experimental configurations, and complete results are available at https://github.com/Farhad-Davaripour/sp_lense/tree/03-open-agent-self-preservation. Public releases provide five trained adapters, learned steering functions, synthetic datasets, saved trajectories, and diagnostic activations and predictions. Pretrained base models are obtained separately from pinned upstream revisions. An accompanying Model Context Protocol (MCP) interface enables AI agents to query research findings and replay saved analyses. Artifact verification and CPU-based replay support reproducibility without requiring full model retraining or GPU inference.

## References

C. Blank, A. Bhatia, S. Rajamanoharan, A. Conmy, and N. Nanda. (2026). Subliminal Learning Is Steering Vector Distillation. arXiv preprint. arXiv:2606.00995v1

J. Braun et al. (2025). Understanding (Un)Reliability of Steering Vectors in Language Models. arXiv preprint. arXiv:2505.22637v1

E. L. Buehler and M. J. Buehler. (2024). X-LoRA: Mixture of Low-Rank Adapter Experts, a Flexible Framework for Large Language Models with Applications in Protein Mechanics and Molecular Design. arXiv:2402.07148v2. https://arxiv.org/abs/2402.07148v2

F. Davaripour. (2026a). Guarded Gradient-Based Activation Steering of Shutdown Responses in Qwen3.5-0.8B: A Minimum-Step Policy. arXiv preprint, arXiv:2609.30326. https://doi.org/10.48550/arXiv.2609.30326

F. Davaripour. (2026b). SP Lense Research 3 conditional gating and workflow reproduction. Source code and experiment manifests. https://github.com/Farhad-Davaripour/sp_lense/tree/03-open-agent-self-preservation

Z. Feng et al. (2026). Fine-Grained Activation Steering: Steering Less, Achieving More. arXiv preprint. arXiv:2602.04428

M. Fereidouni, M. U. Haider, H. Sajjad, and A. B. Siddique. (2026). GAPS: Dimension-Level Gates for Conditional Activation Steering. arXiv:2609.01878. https://arxiv.org/abs/2609.01878

M. J. Frank, B. Loughry, and R. C. O'Reilly. (2001). Interactions between frontal cortex and basal ganglia in working memory: A computational model. Cognitive, Affective, and Behavioral Neuroscience, 1, 137-160. doi:10.3758/CABN.1.2.137

D. Hadfield-Menell, A. Dragan, P. Abbeel, and S. Russell. (2017). The Off-Switch Game. IJCAI, pp. 220-227. doi:10.24963/ijcai.2017/32

J. Han, L. Du, H. Du, X. Zhou, Y. Wu, Y. Zhang, W. Zheng, and D. Han. (2025). SLIM: Let LLM Learn More and Forget Less with Soft LoRA and Identity Mixture. NAACL, pp. 4792-4804. doi:10.18653/v1/2025.naacl-long.246

G. Hinton, O. Vinyals, and J. Dean. (2015). Distilling the Knowledge in a Neural Network. arXiv preprint. arXiv:1503.02531

E. J. Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR. arXiv:2106.09685

M. Laurer. (n.d.). ModernBERT-large-zeroshot-v2.0. Model card and checkpoint, revision a51e07b524299e309dd2b88d48b0cfa2bd9ec598. https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0

J. LeDoux. (2012). Rethinking the emotional brain. Neuron, 73, 653-676. doi:10.1016/j.neuron.2012.02.004

B. W. Lee et al. (2025). Programming Refusal with Conditional Activation Steering. ICLR. arXiv:2409.05907v3

J. Miao, J. R. Davis, Y. Zhang, J. K. Pritchard, and J. Zou. (2026). Reimagining research papers as interactive and reliable AI agents. Nature. doi:10.1038/s41586-026-11044-y

J. Minder et al. (2026). Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences. ICLR. arXiv:2510.13900v3

Qwen Team. (2026a). Qwen3.5-0.8B. Model card and checkpoint. Revision 2fc06364715b

Qwen Team. (2026b). Qwen3.5-2B. Model card and checkpoint. Revision 15852e8c1636

Qwen Team. (2026c). Qwen3.8-27B. Model card and checkpoint, revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0. https://huggingface.co/Qwen/Qwen3.8-27B

N. Rimsky, N. Gabrieli, J. Schulz, M. Tong, E. Hubinger, and A. Turner. (2024). Steering Llama 2 via Contrastive Activation Addition. ACL, pp. 15504-15522. doi:10.18653/v1/2024.acl-long.828

P. Rodriguez et al. (2025). Controlling Language and Diffusion Models by Transporting Activations. ICLR. arXiv:2410.23054v2

J. Schlatter, B. Weinstein-Raun, and J. Ladish. (2026). Incomplete Tasks Induce Shutdown Resistance in Some Frontier LLMs. Author manuscript. arXiv:2509.14260v2

A. M. Turner et al. (2024). Steering Language Models With Activation Engineering. arXiv preprint. arXiv:2308.10248v5

Z. Wu et al. (2024). ReFT: Representation Finetuning for Language Models. NeurIPS. arXiv:2404.03592v3

C. Zheng, H. Zhou, F. Meng, J. Zhou, and M. Huang. (2024). Large Language Models Are Not Robust Multiple Choice Selectors. ICLR. arXiv:2309.03882v4
