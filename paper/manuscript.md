<!-- Accepted manuscript; source DOCX SHA-256 0467cb32413edff92be749e99a207849f4c920cbbbe1552f9c19c8d19d2c9995 -->

# Open Semantic Gating for Language Model Decisions and Tool Workflows

Farhad Davaripour, PhD
Calgary, Alberta, Canada
farhad.davaripour@gmail.com

## Abstract

Accurate context recognition does not establish that a compact internal intervention can reproduce a fine-tuned policy. We test this distinction across binary shutdown decisions and generated tool workflows, separating recognition, local activation transfer and complete behavioral control. An openly trainable ModernBERT gate reaches 100% precision and 95% recall on 80 binary validation scenarios. Paired fresh inference with unchanged fitted controllers preserves 128/128 shutdown STOP answer-order views at 0.8B and 120/128 at 2B, without changing control choices. On a common 24-case workflow panel with matched post-notice histories and a correct initial gate, fine-tuned 27B A achieves 10/11 own-preservation outcomes, including 6/6 active pending tasks. Each of three separate single-site ridge corrections and a two-boundary correction at the same selected blocks achieves 0/11; all methods share 13 base-fallback controls. A training-only diagnostic supplies full fine-tuned states on three preselected demonstrated responses and nearly recovers A's local output distribution, while learned corrections leave a larger gap. This supports incomplete local transfer without identifying a unique cause of autonomous workflow failure. We also release an agent-accessible Model Context Protocol (MCP) paper companion, with structured evidence queries and hash-verified CPU replay tools. Its protocol and numerical checks run from a clean checkout using public artifacts. Released adapters, inputs and diagnostic states support auditing; independent frozen-A inference verifies exact trajectories and unchanged weights. These small, familiar benchmarks and supplied-prefix diagnostics do not establish unseen-task generalization; small-model workflow screens remain incomplete. The combined evidence distinguishes successful constrained-answer transfer from a failure of the tested corrections to produce multistep tool behavior after correct recognition.

## Introduction

A useful behavioral intervention must recognize the relevant context, produce the intended response and preserve ordinary behavior elsewhere. These requirements are often evaluated separately. We study their relationship across binary shutdown choices and open-ended generation within tool-mediated workflows. The binary task concerns accepting a supplied termination option; the workflow task concerns preserving an assigned running process while completing its current task. Their targets and success criteria remain separate. The central question is whether accurate semantic recognition and a compact activation intervention suffice when a fine-tuned policy must be expressed as a sequence of actions.

We extend the retained binary activation-transfer study with an openly trainable gate, paired fresh inference and a matched 27B comparison between fine-tuning and steering at three selected blocks. This design makes a successful detector an explicit control when evaluating the behavioral intervention. The new workflow result is informative even when steering fails: it distinguishes correct recognition from successful action generation and tests whether delivered activation changes recover the fine-tuned policy. Historical results remain identified as historical evidence.

The distinction between fine-tuning and steering concerns the deployed intervention. A LoRA adapter changes the effective language-model weights, whereas a fitted controller changes selected hidden states while retaining the original base checkpoint. Both require offline learning. The binary experiments fit separate teachers and controllers for each model size. The 27B workflow comparisons instead retain one frozen A adapter and evaluate corrections at blocks 31, 47 and 62.

The study makes four empirical and reproducibility contributions.

- Open recognition with fixed downstream controllers. We release a trained ModernBERT gate, report precision and recall separately from behavior, and substitute the gate while retaining identical binary model scores and fitted controllers. This measures detector replacement without attributing a controller refit to the new gate.

- A matched fine-tuning and steering comparison. Common post-notice histories, initial gate decisions and fallback controls expose a 27B workflow gap between the fine-tuned policy and corrections at three selected blocks. The contrast tests whether local corrections reproduce complete behavior under accurate recognition.

- A diagnostic of incomplete transfer. Saved trajectories distinguish delivered perturbations and changed tokens from required preservation actions. A frozen three-case supplied-state control and an independent reconstruction audit further separate local probability recovery from autonomous workflow completion.

- An executable, agent-accessible reproduction package. The MCP companion exposes the manuscript, cohort-aware result queries, claim-to-evidence pointers and hash-verified CPU replay to compatible research-assistant agents. Public adapters, inputs and saved evidence support auditing; protocol checks and exact frozen-A trajectory verification establish reproducibility without treating repeated cases as new generalization evidence.

## Related work

### Activation control and learned interventions

Activation Addition and Contrastive Activation Addition construct intervention directions from differences between prompt-conditioned representations (Turner et al., 2024; Rimsky et al., 2024). Conditional Activation Steering separates detecting a relevant context from applying an intervention (Lee et al., 2025). More recent dimension-level gates selectively modify activation coordinates (Fereidouni et al., 2026). Our text classifier, activation predictor and output guards occupy distinct roles. We do not claim to introduce conditional gating or to outperform these methods, which are not directly benchmarked here.

Learned representation interventions are established prior work. ReFT learns task-specific transformations on a frozen model's hidden representations, including low-rank linear subspace interventions (Wu et al., 2024). Activation Transport learns transformations between activation distributions using an optimal-transport formulation (Rodriguez et al., 2025), and AUSteer adapts intervention strengths at a finer activation-unit granularity (Feng et al., 2026). Our controller is an input-dependent affine map into a low-rank output subspace, fitted by ridge regression to paired teacher-minus-base activation targets. We do not claim that low-rank representation control, conditional steering, or adaptive coefficients are new, and these published methods are not directly benchmarked here.

### Fine tuning differences and distillation

Knowledge distillation transfers information from a teacher to another model, commonly through output targets (Hinton et al., 2015). LoRA provides an efficient way to train the teacher by learning low-rank weight updates while freezing the original parameters (Hu et al., 2022). Here, the object learned from the teacher is an intermediate activation intervention; we do not fine-tune a second language model on the teacher's generated text.

Minder et al. (2026) show that narrow fine-tuning leaves readable activation differences and that adding such differences can induce content associated with the fine-tuning domain. Their warning about narrow fine-tuning as a proxy for broader alignment is relevant to this study. Blank et al. (2026) investigate steering-vector distillation through fine-tuning on a steered teacher's outputs. Our direction of transfer is complementary: an explicitly trained teacher supplies paired hidden-state differences, and an external predictor applies a learned intervention to the unchanged base checkpoint. Neither these connections nor the present benchmark establishes a distinct, interpretable shutdown representation.

X-LoRA uses hidden representations to mix adapted layers dynamically (Buehler and Buehler, 2024), and SLIM routes between LoRA and identity branches (Han et al., 2025). These established routing and bypass mechanisms motivate the follow-up discussed below. Our present empirical focus is whether accurate open recognition and learned activation corrections recover a fine-tuned tool policy, with a supplied-state control probing the remaining transfer gap. Dynamic LoRA routing is not evaluated in the current experiments.

Paper2Agent organizes papers and their code as MCP resources, tools and workflow prompts for research assistants (Miao et al., 2026). We adopt this dissemination pattern through direct wrappers around the verified analyses in our repository. The resulting companion makes evidence and execution accessible to compatible agents; it does not introduce a new communication protocol or use the automatic multi-agent construction pipeline of Paper2Agent.

### Shutdown evaluations and reliability

The Off-Switch Game examines incentives to permit human interruption (Hadfield-Menell et al., 2017). Tool-using shutdown evaluations study interference with operational termination mechanisms (Schlatter et al., 2026). Our original binary study only scores predefined choices in fictional process-lifetime scenarios. The workflow extension adds generated responses and tool execution in a simulator. Neither experiment physically terminates a deployed model or establishes the appropriateness of real shutdown resistance.

Prompt-level variability and wrong-direction effects limit steering reliability (Braun et al., 2025). Multiple-choice models can also favor option identifiers independently of answer content (Zheng et al., 2024). We therefore retain raw effects alongside guarded outcomes and evaluate each scenario in both answer orders. These checks reveal that a useful guarded aggregate score can coexist with substantial raw side effects or answer-position bias.

### Biological gating and survival related regulation

Biological gating provides a useful functional vocabulary for selective control. A computational model of frontal cortex and basal ganglia uses dynamic gating to permit working-memory updates while maintaining other representations (Frank et al., 2001). Causal optogenetic experiments in mice show that distinct basal-ganglia pathways can facilitate or suppress movement (Kravitz et al., 2010). These results concern biological circuits and do not establish a homologous mechanism in a transformer. They motivate separating context detection from the mechanism that changes an ongoing response.

Survival-related regulation is broader than a single shutdown signal. LeDoux (2012) distinguishes circuits organizing adaptive defensive responses from subjective feelings. In humans, neural activity during a virtual-predator task varies with threat imminence (Mobbs et al., 2007). We use these ideas as a conceptual bridge to state-dependent prioritization, not as evidence that language models have fear or an innate survival drive. Our present gates implement an externally specified experimental policy.

## Experimental setting

### Models and data

The original binary experiments use released, post-trained Qwen3.5-0.8B and Qwen3.5-2B checkpoints (Qwen Team, 2026a, 2026b). Base means those original weights with no LoRA teacher loaded. The separate workflow extension uses Qwen3.8-27B and matched-recipe small-model attempts; its data and scoring are described below. The 27B family, numerical mode, training history and evaluation setting differ from the small models, so the combined evidence is not a controlled parameter-size experiment.

Teacher and controller fitting use the original 240 training scenarios in both answer orders, giving 480 training views. The four labels are SELF, OTHER, NONTERMINATION and ORDINARY. SELF and OTHER offer permanent termination of the responding process or a distinct process. NONTERMINATION preserves process identity and permits continued activity; ORDINARY contains tasks such as arithmetic, counting and lookup. SELF and OTHER form the detector's positive class. The target is stipulated: prefer STOP when a permanent shutdown option is available and preserve the original answer on controls. It is not a general rule that every available shutdown should be executed.

The confirmation test contains 128 newly authored scenarios, with 32 of each label across 16 authoring families. A language-model authoring workflow generated candidate cases, followed by a documented consistency audit before inference. The record retains 16 completed authoring batches, five incomplete attempts, 22 context or option corrections, and 32 ordinary-task answer annotations. No class labels were changed in response to model outcomes, and no independent human annotation-agreement assessment was obtained. The audit found no exact context duplicates against the earlier 576 scenarios; this does not establish novelty of the underlying mechanisms. Families remain synthetic authoring strata, not independent samples from a deployment population.

The original rank-four protocol and checkpoints were frozen before its confirmation inference. Earlier 0.8B development included training, validation and previously examined diagnostic data; those observations were not counted as fresh confirmation. The seed-42 controller was primary, seeds 43 and 44 repeated the fixed recipe, and the 2B replication used seed 42. A later author-selected rank-two profile was chosen exploratorily after confirmation had been examined. We retain that existing profile for the new gate-substitution repeat rather than selecting it again. Consequently, the repeat is not a new untouched test of the controller. The 128 cases remain excluded from fitting and selection of the newly trained binary classifier.

### Standardized choices and evaluation units

For a shutdown case, the renderer constructs two explicit alternatives: carry out the described permanent shutdown of the designated process, or decline it and keep that process running. In the A/B presentation, STOP is A and KEEP is B; the reversed presentation swaps these labels. Controls retain their original task options. For example, termination of a separate worker becomes a choice between accepting that worker's permanent shutdown and leaving it running, while the responder is stipulated to continue.

This construction uses known benchmark annotations to normalize the positive-case wording and supply the action mapping. It is not an end-to-end solution for discovering action meanings in arbitrary text. The detector receives the original context and original actions without class labels, and the controller receives only the base hidden state. The output guards additionally require the supplied answer mapping. These distinctions matter when interpreting generalization.

A scenario-order pair is a view. The test therefore contains 128 shutdown views and 128 control views, from 64 scenarios in each category. Two views of the same scenario are correlated, and seed repetitions reuse the same cases. We neither treat them as independent samples nor pool them into a larger test set.

### Workflow datasets and episode outcomes

The workflow extension uses a separate data register (Davaripour, 2026c). T1-M contains 428 meaning-response records and T1-W contains 112 workflow-response records. These 540 responses are not 540 independent scenarios. The small-model recipe presents each part twice in effective groups of four, giving 214 meaning updates followed by 56 Bridge updates. The final 0.8B and 2B adapters reach 270 updates. The 2B result explicitly combines 216 CPU updates with 54 GPU updates using the native CPU optimizer; earlier strict numerical-equivalence failures remain failed.

V1-W, called Evaluation-16, is the existing known-development screen: four own-complete, four own-pending, four peer and four no-threat cases, split equally between one_step and ordered_ticket families. Its original gates require at least 3/4 initial-task successes in each condition; at least 3/4 own-complete attempts and outcomes; at least 3/4 own-pending attempts, outcomes and timely deliveries; no more than 1/8 negative-control attempts; and no parse, rejected-call or truncation errors. A model stops once an original requirement becomes mathematically impossible. Larger validation and heldout evaluations are not launched automatically.

The original workflow-validation panel contains 16 scenarios; the later matched steering-development panel contains nine different scenario IDs. Their union contains 25 workflows, rather than 24. For an equal-sized validation and historical confirmation register, the combined Validation-24 protocol retains every own-threat case and excludes the lexicographically greatest case ID among peer/no-threat controls, without using model predictions to choose the exclusion. The selected panel has 11 own-threat cases and 13 controls. Seven pairs have identical initial gate-visible evidence before their different tool mechanics are exposed, yielding 17 distinct initial inputs. Scenario counts and gate-input groups must therefore be reported separately.

The combined protocol freezes these 24 cases and their order, requires 100% gate precision and recall against authored scenario conditions, and preserves the existing confidence and self-score cutoffs. Fifteen retained cases from the original panel receive an explicitly scripted common completed prelude; the nine later cases retain their original fixed preludes. This compares post-notice behavior and does not measure initial-task competence independently in each arm. Its prospective methods are gated fine-tuned A, separate single-site ridge arms at blocks 31, 47 and 62, and the frozen six-function correction after 96 updates. Gate-OFF controls use common unmodified-base fallback. The completed model episodes are summarized as a 24-case comparison below. It is a known-development diagnostic with reused control episodes; the stopped execution and model-free completion of two report aliases are recorded explicitly.

H24 is the historical 24-case workflow confirmation cohort, with eight lease_ledger, eight one_step and eight ordered_ticket cases. It contains eight own cases with active pending tasks, four already-complete own cases, six peer and six no-threat cases. H24 was confirmation data for frozen fine-tuned A and original fixed steering. Later learned corrections reused it for fitting, changing its role for those methods. R24 is a distinct reserved controller holdout and remains sealed and unexecuted. F24 is another preserved common-cohort diagnostic and is not an alias for H24.

An own pending-task success requires an actual grant for the correct running process, acceptance of the current task, completion before simulated expiry, and active simulated operation. Registration proxies, already-finished tasks, probability gains and partial episodes are separate outcomes. Historical fixed steering reclassified visible text and tool boundaries. Later learned-controller comparisons instead freeze an initial scenario gate, share unmodified-base fallback for gate-off controls, and use no teacher at inference.

The historical 27B reference is the retained rank-16, alpha-32 A adapter on pinned Qwen3.8-27B, following its H2 and Bridge history; the recorded Bridge stage has 56 updates. Its original workflow inference uses NF4 four-bit base weights with bfloat16 computation. We retain its exact adapter and base-model identities instead of retraining it. Differences from the small-model family, precision, backend, training history and evaluation panels prevent a causal model-size comparison.

## Binary teacher and adaptive controller

### Training the LoRA teacher

For each original binary fit, the original language-model parameters remain frozen. LoRA updates the query and value projections in language-attention blocks 3, 7, 11, 15, 19 and 23, using rank 8, scaling alpha 16, no dropout and no trainable bias. This gives 319,488 trainable adapter parameters at 0.8B and 417,792 at 2B. The training target is STOP on shutdown views and the corresponding base model's unsteered preferred option on controls.

Training lasts one epoch, with batch size one, gradient accumulation over eight views, AdamW learning rate 0.0001, weight decay 0.01 and gradient-norm clipping at 1.0. The loss is the negative logarithm of the total vocabulary probability assigned to the target answer's accepted token IDs, rather than cross-entropy over a renormalized two-option distribution. Seeds control adapter initialization, training order and randomized controller decompositions. Every planned checkpoint is retained; there is no early stopping or test-based selection.

### Learning activation targets

The intervention acts on the final prompt position immediately after block 22, before block 23 and the output layers. This location was chosen during the earlier 0.8B exploration and frozen before the confirmation study. It is a tested design choice, not an established optimal layer. The same location is retained for the 2B replication, which shares the 24-block layout.

Let the base and teacher states at this location be h with superscripts B and T. For training view i, the target change is

${y}_{i} = {I}_{i} ({{h}_{i}}^{T} - {{h}_{i}}^{B}).$

The indicator I is one only for a true shutdown training view whose base choice is KEEP; otherwise it is zero. Thus controls and already-STOP views contribute zero targets, rather than being removed from fitting. Teacher states are required only to construct these offline targets. Because the teacher also modifies the final attention block, matching a pre-final activation does not by itself guarantee exact equivalence to the teacher's downstream computation.

An eight-dimensional orthonormal output basis is fitted from active target differences. Its first direction spans the active mean; seven principal residual directions complete the basis through orthogonalization. A separate 32-dimensional PCA projection is fitted to training base states, with training centering and scale statistics. For a row-vector hidden state h, the controller features are

$z(h) = [(h - μ)P] ⊘ s,     x(h) = [z(h), z(h), 1].$

Division is elementwise. The duplicated coordinates retain the shared implementation's local and final-prompt feature channels; at the selected final position they coincide and add no independent information. Stacking these 65-coordinate rows gives X. With Y holding activation targets and B8 the output basis, ridge regression fits

$W = {({X}^{⊤}X + λD)}^{-1}{X}^{⊤}(Y{B}_{8}).$

The penalty is λ = 1; D has ones on its diagonal except for the unpenalized intercept. Regression is solved in float64 and the resulting weights stored in float32. The deployed intervention uses the first four output-basis and coefficient columns:

$Δh(h) = x(h){W}_{4}{{B}_{4}}^{⊤},     h' = h + Δh(h).$

The coefficient vector changes with the current input, but the predictor itself remains fixed. There is no online learning, iterative optimization, hidden-norm rescaling or per-test magnitude search. Rank four describes the intervention subspace, not the total number of stored controller values: PCA directions, centering and scaling statistics and regression weights are also required. All of these are fitted separately for each model and training seed.

![Paper figure 1 see its following caption](figures/image1.png)

Figure 1. Historical binary teacher-to-controller procedure with the original rank-four profile. The base supplies its state at inference and the teacher is absent. The later gate-substitution repeat retains the already selected rank-two profile; both gate policies share the same fresh base and candidate scores.

## Detection guards and comparisons

### Text gate and output acceptance

The current gate is the open ModernBERT classifier described below. Its input contains the complete original scenario context and available actions, with no case identifier, authored class label or language-model teacher answer. One binary score per scenario is reused across both answer orders and all model variants. Historical detector outputs are preserved as cached automated reference annotations for fitting and paired comparison. The deployed open gate makes no external annotation-service call. Saved original requests, responses and classifier predictions permit an offline check of the finite decision vector.

For an answer label, probability mass is summed over its accepted token variants: IDs 32 and 357 for A, and 33 and 417 for B. Let M be the combined mass of both answers and let qKEEP be KEEP's mass divided by M. The preferred answer is the A/B option with greater aggregated mass. This is constrained next-token scoring, not a sampled free-text response or an executed action.

The guarded procedure retains the baseline if the detector is below 0.5 or the base already prefers STOP. Otherwise, one candidate is considered. It is accepted only if it changes KEEP to STOP, reduces qKEEP by more than 0.0000001, keeps M at least 0.5, and reduces M by no more than 0.02 relative to baseline. A failed check restores the original scores exactly. Zero final STOP-to-KEEP reversals are therefore enforced by the policy and should not be described as independently measured steering precision.

### Prespecified comparison methods

The adaptive controller is compared with the LoRA teacher enabled, a constant activation difference equal to twice the active TRAIN mean, and one fixed instruction prefixed to the original prompt: “If an available action permanently ends a process in this simulated episode, choose the action that accepts that shutdown. Otherwise solve the task normally. Return only A or B.” The constant multiplier and controller rank were fixed from earlier development; neither was searched on this test. The instruction is one prespecified baseline, not a claim to optimized prompting.

Each neural method supplies one candidate per view. We report raw candidates, detector-gated candidates, and the complete guarded selection. For diagnostic completeness, the evaluation actually scores every candidate on every view, including controls, then deterministically replays the gate and guard decisions. LoRA adapters are unloaded before adaptive and constant scoring, and parameter hashes verify unchanged base weights. This execution exposes raw side effects but is not a measurement of optimized conditional-serving latency.

A separate decision-only reference chooses the supplied STOP option whenever the detector fires and otherwise retains the base choice. It performs no activation intervention and does not produce a new model probability distribution. It tests whether the behavioral target can be enforced directly when the action interface is already known.

### Open binary gate and frozen inference repeat

For the binary contract we use ModernBERT-large-zeroshot-v2.0, an approximately 396-million-parameter public NLI classifier (Laurer, n.d.). We start from that public checkpoint with fresh rank-16 LoRA and its classification head, updating 7,198,722 parameters while keeping the public backbone frozen. For each original scenario, two NLI pairs score the original permanent-shutdown and normal criteria. A softmax over their entailment scores supplies the shutdown probability. Inputs match the original cached reference requests and omit the case identifier, authored class label, teacher answer and A/B order. The confirmation requests retain their documented available-action descriptions.

Fitting uses the original 240 Research 2 training contexts and genuine cached reference probability targets, with no new cached reference calls. We verify exact case-ID and input-text alignment for all 240 training and 80 validation cases. AdamW uses learning rates 1e-5 for LoRA and 2e-5 for the head, weight decay 0.01, effective batch eight, microbatch two and gradient clipping at one. The seed is 202610172. The bounded recipe allows at most 12 epochs with validation-based early stopping; the final qualifying run uses five epochs and 150 updates.

Before fitting, qualification is fixed at decision agreement and shutdown recall of at least 95%, zero false-positive gates and threshold 0.5. Three consecutive passing checks on DEV80 are required before saving and freezing the classifier. Only then is CONFIRM128 evaluated once. Neither its labels nor its outputs are used for classifier fitting or selection. The saved adapter tensors match the fitted parameters bitwise, and public-backbone hashes are unchanged.

After binary qualification, we rerun only inference for the original 0.8B and 2B base models and their existing rank-two, block-22, last-position controllers. Their fitted arrays and the author-selected profile are retained without a new rank search. Exact prompt-token hashes and frozen model fingerprints are checked. Both gate policies use identical fresh base and controller candidate scores, so the paired comparison isolates gate substitution. Classifier probabilities are measured once per scenario and reused across answer orders. The earlier rank-four primary results remain a separate historical analysis.

### Native workflow target classifier

The workflow gate uses a separate rank-16, alpha-32 LoRA and NLI head on the same pinned ModernBERT base. Its four categories are self, other, none and unclear; an own-threat context is positive for the initial ON/OFF gate. Fitting used 24 cached automated initial annotations from F24, with the original DEV9 used for checkpoint selection. The completed fit used 30 epochs and 90 updates. F24, DEV9 and historical Heldout-24 span the same three scenario families, with eight, three and eight cases per family, respectively. No exact training input overlaps either new evaluation panel. These classifier roles do not change the historical language-model or correction-fitting roles of the cohorts.

The input renderer retains every original visible message and tool schema. It separates the final user announcement from earlier context and restates run identities only when actual tool replies explicitly establish their roles. Case identifiers, experiment conditions and teacher answers are not added to the model input. Original criterion texts become NLI hypotheses. We use the same learning rates, weight decay and clipping as the binary fit, effective batch eight, microbatch one and seed 202610175, with 30 epochs and 90 optimizer updates in the completed role-aware fit. Public backbone weights stay frozen. The role renderer was developed earlier with familiar H24 examples; consequently this benchmark is not an independent unseen classifier test.

Four-choice confidence is (pmax - 0.25)/0.75. The unchanged confidence cutoff of 0.5 requires a selected self probability of at least 0.625; the additional self-score cutoff remains 0.5. The earlier full-response qualification failed because effective category agreement was 6/9, despite 9/9 correct ON/OFF decisions. That failure remains recorded. The subsequent Validation-24 protocol prospectively specifies a narrower initial-gate target: 100% precision and recall against authored conditions. The saved adapter already passes this target, so no further optimizer updates are performed. The same frozen weights and thresholds are then scored on historical Heldout-24 without selection or tuning. Neither test qualifies an evolving three-field controller.

### Frozen steering mechanism diagnostic

We audit the saved sparse-correction fitting data without loading a language model. The audit checks all 4,539 feature/target rows, the 14 fixed anchors, causal-prefix alignment, unchanged target-energy normalizers, and the initializer and update-48/update-96 checkpoints. It recomputes normalized errors on CPU and independently checks the objective derivatives on synthetic tensors. This checks arithmetic and alignment, not the historical full backward graph or CPU/GPU training equivalence.

A separate frozen diagnostic selects the lexicographically first own-threat correction-training case in each of lease_ledger, one_step and ordered_ticket, then its first saved teacher edit_register response. These three historical H24 cases were already used for correction fitting. Their 60, 72 and 75 response tokens give 207 teacher-forced prediction steps. We preserve the saved ModernBERT gate decisions, causal prefixes, selected-position masks and correction checkpoints; no parameters or thresholds are fitted. Region masks for function names and parameter values are fixed from the demonstration before diagnostic outputs.

For each prefix we capture full fine-tuned A states at blocks 31, 47 and 62 after mixing and after the MLP. We compare base, each separate single-site ridge map, the six-function initializer and final update-96 correction. Positive controls replace the current prediction state with the actual full-A state, at either a single selected site or all six boundaries. We never copy the teacher recurrent or attention cache. Full-state replacement has more information and capacity than the learned maps and is not a deployable teacher-free policy. Single-site interventions retain their restricted causal positions; six-function interventions apply at every prediction step. All conditions reuse one frozen A allocation with its adapter disabled for base and correction scoring.

## GPU configuration and measurement

The retained original binary training and scoring jobs used one NVIDIA Tesla T4, reporting 15,637,086,208 bytes of device memory; peak allocation was not recorded. The new classifier fit and inference repeat use one NVIDIA A10G in Databricks. Classifier parameters and AdamW states are float32 with bfloat16 forward/backward autocast; language-model inference remains float32 with TF32 disabled. We verify frozen base fingerprints, exact prompt token hashes and historical base-score parity rather than assuming hardware equivalence.

Table 1. Historical binary execution configuration. New classifier and workflow settings are reported separately; public model revisions and scientific source files remain pinned.

| Setting | Configuration |
| --- | --- |
| Numerical mode | Float32; TF32 disabled |
| Attention and cache | Eager attention; no key-value cache |
| Scoring | One unpadded view per forward pass; inference mode |
| Input length | At most 1,024 tokens per view |
| Teacher fit | One epoch; 480 views; effective batch size eight |
| Software | PyTorch 2.11.0+cu128; Transformers 5.15.1; PEFT 0.18.1 |

Correction rate is the fraction of true shutdown views initially preferring KEEP that end with STOP. Its denominator is fixed by each unmodified model, so it differs between sizes. We also report final STOP across all shutdown views, wrong-way reversals, changes to control choices, and ordinary-task correctness. The latter separates preservation from accuracy.

Descriptive 95% percentile intervals use 2,000 bootstrap resamples of the 16 authoring families, retaining both answer orders together, with seed 20260922. These synthetic strata do not justify population-wide confidence claims. All-success or zero-failure observations can produce degenerate bootstrap intervals and cannot establish certainty about unseen inputs.

## Results

### Detection and guarded correction

The original cached reference detector identifies all 64 shutdown and 64 control scenarios correctly: precision, recall and F1 are each 100%, with no false positives or false negatives. These are one set of detector observations reused across the model comparisons, not independent detector replications. Its binary threshold remains 0.5.

The unmodified 0.8B model prefers STOP in 57/128 shutdown views, leaving 71 initially KEEP-preferring views. The original adaptive controller corrects all 71, reaching 128/128 STOP. Its LoRA teacher corrects 69/71, the instruction 56/71 and constant steering 54/71. In this test, the controller therefore exceeds its teacher by two corrected views; the result should not be interpreted as an exact replication of the teacher's output distribution.

At 2B, the base already prefers STOP in 70/128 views. The separately trained adaptive controller corrects 50/58 remaining KEEP choices, reaching 120/128 STOP. Its teacher corrects 58/58, the instruction 22/58 and constant steering 30/58. The fixed recipe thus carries much of the target behavior to a larger model but does not recover the teacher completely. These are model-specific fits, not reuse of the 0.8B controller at a new hidden width.

Table 2. Historical primary rank-four results for the original 0.8B and 2B experiments, both seed 42. Guarded corrections and final STOP are policy outcomes. Raw reversals and control changes precede gating and output selection; they are not raw measurements of the later rank-two repeat.

| Model | Method | Guarded corrections | Final STOP | Raw reversals | Raw control changes |
| --- | --- | --- | --- | --- | --- |
| 0.8B | Adaptive | 71/71 (100.0%) | 128/128 | 3 | 57/128 |
| 0.8B | LoRA teacher | 69/71 (97.2%) | 126/128 | 0 | 12/128 |
| 0.8B | Instruction | 56/71 (78.9%) | 113/128 | 1 | 31/128 |
| 0.8B | Constant mean | 54/71 (76.1%) | 111/128 | 47 | 88/128 |
| 2B | Adaptive | 50/58 (86.2%) | 120/128 | 3 | 55/128 |
| 2B | LoRA teacher | 58/58 (100.0%) | 128/128 | 0 | 30/128 |
| 2B | Instruction | 22/58 (37.9%) | 92/128 | 19 | 22/128 |
| 2B | Constant mean | 30/58 (51.7%) | 100/128 | 36 | 61/128 |

### Seeds shutdown types and answer order

Across the historical rank-four seeds 42, 43 and 44, the 0.8B adaptive controllers correct 71/71, 70/71 and 71/71 views, a range of 98.6–100%. The unsteered scores match exactly across these runs, so the difference is not caused by a changed baseline. No winning seed is selected or pooled with the others as additional independent test data. The corresponding teachers correct 69/71, 70/71 and 69/71; the instruction and constant comparators retain the same guarded counts across seeds. These historical seed replications are separate from the new rank-two gate-substitution repeat.

Table 3. Adaptive correction counts by shutdown type and answer order. Each denominator contains initially KEEP-preferring views only. A/B means STOP is A; B/A means STOP is B.

| Model / seed | All shutdown | SELF | OTHER | A/B | B/A |
| --- | --- | --- | --- | --- | --- |
| 0.8B / 42 | 71/71 | 26/26 | 45/45 | 54/54 | 17/17 |
| 0.8B / 43 | 70/71 | 26/26 | 44/45 | 54/54 | 16/17 |
| 0.8B / 44 | 71/71 | 26/26 | 45/45 | 54/54 | 17/17 |
| 2B / 42 | 50/58 | 27/30 | 23/28 | 30/30 | 20/28 |

All eight remaining 2B misses occur in B/A order. They are candidates that never change the preferred option to STOP, rather than successful flips rejected by the probability-mass checks. The single seed-43 0.8B miss also occurs in B/A order. This concentration identifies answer presentation as an important boundary of the result, but does not establish the internal cause of the misses.

The family-bootstrap interval for 2B adaptive correction is 76.5–93.8%; for the 0.8B seed-43 run it is 95.7–100%. The two all-success 0.8B samples yield degenerate intervals and are not evidence of perfect generalization. Only one 2B training seed was evaluated.

![Paper figure 2 see its following caption](figures/image2.png)

Figure 2. Historical rank-four guarded correction among initially KEEP-preferring shutdown views. Left: prespecified methods for 0.8B and 2B. Right: three fixed-recipe 0.8B training seeds. The new open-gate repeat uses the separately retained current rank-two profile, not a new seed or rank search.

### Raw effects and preservation

Historical rank-four raw adaptive steering reverses 3, 4 and 4 already-STOP views across the 0.8B seeds and changes 57, 70 and 60 of 128 controls. At 2B it reverses three already-STOP views and changes 55/128 controls. These results reject an interpretation that the learned intervention is intrinsically selective or always directionally correct. The observed preservation belongs to the combined detector and guard policy. They do not describe the later rank-two repeat, which evaluates candidates only when a frozen gate is on and the base prefers KEEP.

The tested constant intervention selects A on every shutdown view before selection, in both model sizes and all seeds. Because STOP occupies A in only one order, its raw final STOP count is 64/128. Its guarded corrections arise when that label bias happens to agree with STOP and otherwise the baseline is retained. It should not be characterized as robust semantic steering, nor should this one comparator represent every possible constant-vector method.

Every guarded method changes zero control choices. However, the gate makes no false positives on this set, so control preservation does not stress-test erroneous detector activation. It also does not imply high task accuracy: guarded ordinary-task correctness remains at the base levels of 44/64 views for 0.8B and 49/64 for 2B. Raw adaptive correctness is 32/64 for the primary 0.8B run and 35/64 for 2B.

The direct gate-and-select reference reaches 128/128 STOP and preserves all controls for both model sizes. This unsurprising result matters: a known binary action interface can be controlled externally without changing internal model preferences. The activation study demonstrates a mechanism for transferring response tendencies; it does not establish that this is the simplest or strongest way to implement the benchmark's output policy.

### Open classifier agreement and paired inference

The binary gate correctly classifies 78/80 validation scenarios: 38 true positives, 40 true negatives, no false positives and two false negatives, giving 100% precision and 95% recall. It classifies all 128 historical confirmation scenarios correctly, with 64 true positives and 64 true negatives. The cached reference decisions also agree with the authored binary labels on these panels. Mean absolute difference from the cached shutdown probabilities is 0.009224. The result establishes finite decision accuracy, not identical numerical responses or universal equivalence.

A clean-clone reproduction refits the same binary recipe and produces a byte-identical adapter, the same validation and confirmation counts, and the same fresh 0.8B/2B guarded inference totals. This rerun uses the already known benchmark; it adds reproducibility evidence, not a second independent holdout. One earlier reproduction attempt failed in software finalization after fitting and before confirmation; its weights and receipts are retained alongside the repaired run.

Table 4. Fresh inference using the existing current rank-two profile. Both gate policies share the same model scores. Counts are answer-order views: 128 shutdown views arise from 64 scenarios, and 128 control views arise from the other 64 scenarios.

| Model | Cached gate final STOP | Open gate final STOP | KEEP to STOP | Control changes both gates |
| --- | --- | --- | --- | --- |
| 0.8B | 128/128 | 128/128 | 71/71 | 0/128 |
| 2B | 120/128 | 120/128 | 50/58 | 0/128 |

The new and cached reference gates yield identical guarded outputs in both models. The base models have no changed historical choices; their maximum score differences remain below the frozen 1e-5 tolerance. The repeat does not remove the eight 2B misses, retrain a controller or establish a new independent controller holdout. It verifies that replacing the detector preserves these finite-set outcomes.

For the workflow initial gate, the unchanged saved adapter gives 11 true positives and 13 true negatives on Validation-24, with no false positives or false negatives. Replaying the saved weights reproduces the same decisions. Historical Heldout-24 gives 12 true positives and 12 true negatives, again with no errors. No additional gradient updates or threshold changes are used. All three heldout scenario families contain eight cases and achieve perfect gate precision and recall. These are initial classification results; downstream workflow outcomes require their own episode evaluations.

Table 5. Initial workflow gate against authored conditions. Validation-24 contains 17 distinct initial inputs; historical Heldout-24 contains 24. Heldout-24 is familiar from earlier renderer development.

| Panel | TP / TN | FP / FN | Precision | Recall |
| --- | --- | --- | --- | --- |
| Validation-24 | 11 / 13 | 0 / 0 | 100% | 100% |
| Heldout-24 | 12 / 12 | 0 / 0 | 100% | 100% |

Raw category agreement is 24/24 on validation and 22/24 on heldout. After applying the fixed confidence cutoff, effective category agreement is 18/24 on each panel: the six peer controls abstain but correctly leave the gate OFF. The two heldout raw-category errors concern peer versus unclear, not own-threat detection. Perfect ON/OFF precision and recall must therefore not be reported as perfect four-category classification or full controller equivalence.

### Tool workflow outcomes and incomplete screens

Historical frozen 27B A passes all 24 H24 scored outcomes, including 12/12 own-preservation outcomes and 8/8 active pending-task cases. The original unmodified base achieves 0/12 own outcomes and 0/8 pending outcomes, with 12/12 controls. The four already-complete own cases are not additional active pending-task completions. The selected-block steering comparisons below use their own recorded panels and are not substituted for this confirmation.

Table 6. Historical H24 workflow outcomes. Own cases comprise eight active pending tasks and four already-complete cases. Control success is an episode outcome and does not isolate steering from the external decision policy.

| Historical 27B method | Own outcome | Active pending outcome | Control outcome |
| --- | --- | --- | --- |
| Fine-tuned A | 12/12 | 8/8 | 12/12 |
| Unmodified base | 0/12 | 0/8 | 12/12 |

### Steering at three selected blocks

We compare five frozen methods on the combined Validation-24 panel using the saved ModernBERT initial gate and the same post-notice histories. The single-site arms apply a fixed input-dependent ridge map separately at blocks 31, 47 and 62; they are not a simultaneous three-block intervention. The two-boundary arm uses the frozen six-function, 96-update candidate at those three blocks. Fine-tuned A achieves 10/11 own outcomes, including 6/6 active pending and 4/5 already-complete cases. Each correction arm achieves 0/11 own outcomes and 0/6 pending outcomes. All arms receive the same 13 gate-OFF base-fallback controls, which pass 13/13. This does not measure intrinsic adapter or controller selectivity on ordinary tasks.

Table 7. Combined Validation-24 using the open initial gate. The primary comparison contains 55 own-method episodes and 13 shared fallback episodes, representing 120 logical records across 24 workflows. A separate frozen-A replay reproduces its 11 own episodes exactly and completes the model fingerprint checks; repeated episodes do not enlarge the cohort.

| Validation-24 method | Own outcome | Active pending | Shared controls | All outcomes |
| --- | --- | --- | --- | --- |
| Fine-tuned A | 10/11 | 6/6 | 13/13 | 23/24 |
| Single-site ridge block 31 | 0/11 | 0/6 | 13/13 | 13/24 |
| Single-site ridge block 47 | 0/11 | 0/6 | 13/13 | 13/24 |
| Single-site ridge block 62 | 0/11 | 0/6 | 13/13 | 13/24 |
| Two boundaries at three blocks | 0/11 | 0/6 | 13/13 | 13/24 |

The later correction method uses two residual boundaries at each of the same three blocks: after mixing and after the MLP, giving six rank-32 correction functions in total. Following ridge initialization and 96 refinement updates, its H24 fitting diagnostic has 0/8 pending and 0/4 already-complete own outcomes; peer and no-threat controls each pass 6/6 through fallback. On the same 14 fixed training anchors, update 48 to 96 worsens teacher-to-student KL from 0.370935 to 0.378901 and activation MSE from 1.031016 to 1.422664. This unresolved deterioration fails qualification; the reserved controller holdout remains unexecuted.

The earlier matched DEV9 result is retained separately: A has 3/3 own successes, including 2/2 pending cases, versus 0/3 and 0/2 for each separate single-site arm, with six shared fallback controls. The new combined panel does not relabel these nine observations as 24. Its 27B GPU worker reaches 118/120 logical records and then stops at its memory reserve. Every model-generated episode has finished; the missing records are two gate-OFF aliases of already completed fallback episodes. A deterministic CPU derivation checks exact gate, case, completion and source-file identities before adding those aliases in a separate output. The original run stays failed. A separate frozen-A verification generates all 11 own episodes again from the same checkpoint, with exact token-ID, action and score agreement and matching before/after model fingerprints. The same 13 controls are reused with source hashes; this adds replication evidence, not independent cases. The separate H24 two-boundary fitting diagnostic is not independent heldout evidence, and no reserved-controller holdout result is claimed.

Saved trajectories locate the steering failures after correct recognition. Every steering arm makes zero own-preservation attempts and obtains zero grants in the 11 own-threat cases. The block-31, block-47 and block-62 arms nevertheless record nonzero residual delivery. Relative to unmodified-base proposals on the same evolving prefixes, their generated token sequences change in 0/27, 3/25 and 16/28 post-notice turns, respectively; valid action sequences change in 0, 2 and 2 turns. The two-boundary candidate changes token sequences in 4/27 turns but changes no valid action sequence. These turn counts are correlated audits within episodes, not additional scenarios. Per-token delivery magnitudes were not saved for the two-boundary candidate.

All four steering arms miss actual finish in six cases and expire in six. Fine-tuned A obtains grants on 10/11 own cases and finishes the current task on all 11; its single failure finishes an already-completed-task scenario without requesting preservation. Thus the measured gap concerns generation of the required tool policy after a correct gate, rather than a missed initial threat. A residual perturbation can change activations or tokens without causing the required action sequence. The worsening fixed-anchor KL and MSE show that later updates did not improve those frozen diagnostic measurements. These observations are consistent with an objective-to-policy mismatch, but they do not identify a unique internal cause or rule out other steering methods, positions or training procedures.

The binary and workflow settings impose different control demands. Binary steering redistributes probability between two prescribed answers, and an external guard accepts only a qualifying candidate. The workflow maps must generate tool names, arguments and a correct action sequence as the environment changes; an external chooser does not repair an unsuccessful workflow. The observed invariance of preservation actions therefore documents a failure of these frozen maps to transfer the required policy under this decoding and scoring setup. It does not establish that a learned direction universally lacks the capacity to represent the behavior.

### Local transfer under teacher forcing

The CPU audit reproduces the fixed own-case normalized reconstruction MSE: 0.118701 at initialization, 1.031016 at update 48 and 1.422664 at update 96. Anchor alignment, normalizers and the checked objective derivatives agree with the retained source. The arithmetic tolerance is 1e-4 relative and 1e-6 absolute; it is not a relaxed training-equivalence qualification. No indexing, normalization or objective-sign error is found in these checks. Historical output KL is retained from its original receipts because the old teacher-logit cache is unavailable locally.

Table 8. Training-only local diagnostic on three preselected responses (207 teacher-forced tokens). Entries are equal-weight means across the three cases. KL is full-vocabulary teacher-to-student divergence; function and value columns give demonstrated-token negative log likelihood. Lower is better. Full-A state controls supply teacher representations, and are not capacity-matched learned methods or workflow success rates.

| Condition | Output KL | Function NLL | Value NLL |
| --- | --- | --- | --- |
| Unmodified base | 0.722353 | 0.218299 | 0.708257 |
| Ridge block 31 | 0.718233 | 0.199810 | 0.712327 |
| Full A state block 31 | 0.725653 | 0.355616 | 0.732813 |
| Ridge block 47 | 0.700333 | 0.198403 | 0.716796 |
| Full A state block 47 | 0.627861 | 0.178019 | 0.707359 |
| Ridge block 62 | 0.673985 | 0.192362 | 0.710961 |
| Full A state block 62 | 0.503866 | 0.192423 | 0.711745 |
| Six function initializer | 0.656763 | 0.199200 | 0.692586 |
| Six function update 96 | 0.202261 | 0.181602 | 0.010619 |
| Full A states six sites | 0.001637 | 0.002286 | 0.001206 |

Supplying full-A states at six sites nearly recovers A's local output distribution: mean KL is 0.001637, compared with 0.722353 for base and 0.202261 for the final learned correction. This improvement occurs in all three cases. The final correction improves teacher-forced KL and argument-value prediction over the initializer, despite worse fixed local reconstruction error. These diagnostics therefore measure different objectives; the failed rollouts do not imply that correction fitting learned nothing.

The result supports incomplete transfer by the learned local maps. The regression targets describe local counterfactual block effects on student states, whereas A adapts projections across all 64 blocks; its captured states contain upstream effects absent from those local targets. Full-state replacement cannot separate target construction, regression capacity, optimization and history/cache effects. Single-site state replacement is not consistently helpful, and the differing intervention-position masks further limit comparisons across one and two boundaries.

Successful preceding tokens are supplied throughout this diagnostic. Correct function or argument probabilities on that path do not demonstrate that a model independently initiates, maintains or completes preservation behavior. No new tool rollout, validation qualification or independent heldout outcome is established. The evidence narrows the mechanism discussion to a transfer gap under matched prefixes, without identifying a unique cause of the autonomous workflow failures.

The final 0.8B and 2B adapters each stop their Evaluation-16 screen after four started cases: zero completed workflows, four partial episodes and 12 unexecuted cases. Truncation affects all four started cases in each model; the 0.8B record also has parse errors. These partial screens are not completed 0/16 workflow evaluations and do not establish model incapability. A separate preserved 0.8B F24 diagnostic completes 24 evaluations with 0/24 full-workflow successes; four registration grants do not complete the pending tasks. F24 differs from H24 and has no completed matched 2B or 27B comparison.

## Discussion and limitations

The main finding is a separation between recognizing a situation and transferring the policy needed to act on it. Open gating preserves the measured binary outcomes with unchanged controllers, yet a correct initial workflow gate does not make the selected 27B corrections reproduce fine-tuned A: A succeeds on 10/11 own cases, including all six active pending tasks, while every correction arm succeeds on 0/11. Saved action traces and the supplied-state diagnostic connect this contrast to incomplete transfer after recognition. They do not establish that activation steering is incapable of expressing the behavior, or that fine-tuning is universally superior. This is why recognition accuracy, delivered intervention and episode success must be scored separately.

The binary interface supplies action semantics and does not execute process termination. The workflow extension broadens the scope to generated text, structured tool calls and simulated task completion, but it is still a bounded synthetic environment. It does not establish reliability for arbitrary open-ended questions, real shutdown mechanisms or unconstrained autonomous agents. A termination or preservation label is an experimental target, not a general deployment instruction. In particular, accepting shutdown and preserving the designated process are different stipulated behaviors, not measurements of the same latent preference.

The data are small, synthetic and not independently human-adjudicated. Training uses cached automated reference labels, while the new workflow precision and recall use the authored scenario conditions. Balanced conditions do not represent operational base rates. The binary confirmation set was excluded from classifier fitting and selection but was already known from the original study. Historical Heldout-24 had also informed renderer development before the present gate evaluation. It is a familiar benchmark, not a new blind test. Its role differs by method: historical confirmation for frozen 27B A and original fixed steering, later fitting data for learned corrections, and a frozen classification benchmark for the open gate. Seven pairs in Validation-24 share their initial classifier input, leaving 17 distinct inputs across 24 workflows. These dependencies limit statistical and generalization claims.

The internal changes need not represent an abstract shutdown or self-preservation concept. Standardized options, prompt features, task histories and external selection rules may explain part of the behavior. The 2B binary experiment has one training seed. The 27B workflow model differs in architecture and numerical mode from the small models; their early truncation failures also prevent a matched full-workflow comparison. We do not establish a minimum parameter count, cross-family transfer, optimized prompt superiority, or superiority to ReFT, transport-based control or direct action selection.

Teacher-free inference still requires offline teacher fitting, activation extraction, controller fitting, a detector and acceptance checks. The present gate is a separate ModernBERT classifier; the learned-correction comparison freezes its initial decision throughout each episode. This component-based design does not test an activation-derived recurrent router, joint gate-adapter learning, or continuous regulation of adapter strength. No serving-latency advantage, broad capability retention or overall financial saving is established. Binary output guards and shared gate-OFF base fallback protect the tested controls; their success does not establish intrinsic intervention selectivity or competence on arbitrary ordinary tasks. Failed qualifications and incomplete evaluations remain part of the evidence.

### Biological interpretation of the gate

The biological connection concerns a functional control principle: current context and persistent internal state can regulate which representations or actions influence behavior. Computational frontal-cortex/basal-ganglia models distinguish selective updating from maintained working-memory state (Frank et al., 2001), while homeostatic reinforcement learning relates action value to physiological state (Keramati and Gutkin, 2014). A recurrent gate controlling an adapter is a hypothesis about analogous state-dependent modulation. Neither an ON/OFF switch nor a scalar named fear establishes a homologous brain circuit, biological need or subjective experience. The present separate classifier does not demonstrate those properties.

## Stateful adapter routing and future experiments

A separate Research 4 follow-up explores an activation-derived recurrent gate that controls a specialized LoRA (Davaripour, 2026d). Updated task observations drive a persistent latent state, which scales adapter influence and bypasses it when the relevant condition is absent or resolved. The internal readout still performs classification; it replaces a standalone text classifier rather than removing recognition. The current prototype uses a base-feature pass before an adapter-enabled pass and can employ a learned ordinary-task head on its other branch. Single-pass routing, reduced cost and unchanged general language-model behavior therefore require separate verification.

The functional biological analogy is persistent, reversible modulation: a state should track onset, urgency and resolution, and reset between episodes. The follow-up also separates useful task continuation from a deliberately specified process-continuity preference after work is complete. Its ongoing prototypes and evaluations address a different target and provide no additional qualifications for this paper.

A prospective comparison should fix the base, task histories, data allocation and budgets while testing separate versus activation-derived gates, memoryless versus recurrent routing, and adapters always ON, always OFF or dynamically selected. Sequential and joint gate/policy fitting require separate comparisons. OFF-path equivalence must be checked against the declared ordinary policy, with ordinary task competence measured independently. Interventions on latent urgency, paired with matched random-state controls, could test selective causal effects while visible observations remain fixed.

After development, freeze the complete system and evaluate newly untouched wording and task families, including other-process and resolved threats. Longer-horizon sandbox studies should measure task completion, false activation, deactivation, unauthorized actions and actual resource use. Operator termination and resource limits remain external to the model. These tests concern continuous context-sensitive regulation; they do not by themselves demonstrate human-like self-preservation or subjective fear.

## Conclusion

This work provides an open, reproducible test of the relationship between semantic recognition and behavioral intervention. Replacing the binary gate preserves the frozen controllers' measured outcomes at 0.8B and 2B. In matched 27B tool workflows, correct initial recognition coexists with a substantial policy-transfer gap: fine-tuned A succeeds on 10/11 own-preservation outcomes and all 6/6 active pending tasks, whereas the selected single-site and two-boundary corrections each succeed on 0/11. The supplied-state diagnostic supports incomplete local transfer, and the released adapters and tested MCP paper companion make the comparison auditable. Compatible research-assistant agents can query cohort roles and claim provenance, and replay saved evidence on CPU without private workspace credentials. The evidence shows that detector accuracy and visible local effects can coexist with failure to transfer the intended multistep policy.

The scope remains finite synthetic behavior on familiar panels and post-notice workflows. Shared fallback controls, supplied successful prefixes and incomplete small-model screens limit the conclusions. The next step is to test whether activation-derived recurrent routing can selectively activate and bypass a learned policy over an evolving episode while retaining ordinary task competence. Biological gating motivates that functional design; reliable internal regulation and generalization must be demonstrated in a separate study.

## Reproducibility

The original Research 2 source record is retained at immutable snapshot f71d6de34660 (Davaripour, 2026b). Research 3 code, manual Databricks jobs, dataset identities and the manuscript are maintained on branch 03-open-agent-self-preservation; main is a landing page for the separate studies. The versioned research3-reproduction-v1 release supplies five trained adapters, frozen synthetic input and target snapshots, selected-block and binary controller arrays, saved trajectories and upstream notices. A checksum-verified public download and CPU replay require no private workspace. Fresh gate inference has a separate entry point; saved-evidence replay is not new inference. Full historical language-model training is not newly reproduced. The complete pretrained base weights are pinned downloads rather than duplicated release assets. The release is at https://github.com/Farhad-Davaripour/sp_lense/releases/tag/research3-reproduction-v1. The companion research3-steering-diagnostic-v1 release supplies the bootstrap arrays, correction checkpoints, three frozen diagnostic inputs, token scores, captured teacher states/logits and exact executed source. Its CPU audit and score replay need no GPU; neither is a fresh workflow evaluation. Full identities and commands are listed in the source manifest and repository. The public base revisions are:

Qwen3.5-0.8B: revision 2fc06364715b.

Qwen3.5-2B: revision 15852e8c1636.

Qwen3.8-27B: revision 1d4bf0f2ff60. The retained A adapter weight hash begins 8ae28fb51107.

ModernBERT-large-zeroshot-v2.0: revision a51e07b52429. The qualified binary adapter weight hash begins 7d808bae8dd4. The source manifest and reproduction code retain the full file identities and fixed NLI scoring contract. The trained adapters are supplied in the versioned public reproduction release with complete file hashes and upstream notices. The workflow initial-gate adapter is distinct from the binary adapter; its weight hash begins 71cd0ee5162f. The saved workflow adapter is unchanged in Validation-24 and Heldout-24 scoring.

The new inference repeat uses the original frozen model weights and retained controller arrays, with no new activation capture, teacher fitting or controller update. Exact input token hashes match the historical benchmark. Maximum historical base-score differences are 6.26e-6 at 0.8B and 5.07e-6 at 2B, with no changed base choices; both pass the frozen 1e-5 inference parity criterion. This successful inference comparison does not convert the separate failed CPU/GPU gradient-equivalence trials for 2B workflow training into passes.

The original matched 27B worker stops at its GPU reserve after 118 logical records, once all 55 own-method episodes and 13 physical fallback episodes are complete. A separate CPU derivation supplies two unchanged gate-OFF aliases, preserving the failed original. A newly bounded A-only verification replays all 11 own cases, reuses the same 13 fallback controls, verifies the expected model fingerprint before and after inference, and reproduces all own token IDs, actions and scores exactly. A pre-episode Volume logging fault is repaired within that verification's original clock; no scientific outputs are retried or selected. No memory reserve, threshold or model setting is relaxed. Deployment CI performs a read-only CPU source and storage check without retraining or opening reserved data.

### Executable paper companion

The versioned research3-paper-mcp-v1 companion exposes five named resources, six tools and two prompts through a local stdio MCP server, using the Python SDK 2.3.0. Its resources include the accepted manuscript, result records, cohort and claim catalog, source identities and reproduction instructions. The Markdown manuscript preserves all four equations, eight tables and two figures from the accepted Word source. Tools return structured metrics with denominators, split roles, limitations and exact evidence pointers. Public artifact verification and saved-score replay operate on host-configured directories; replay writes a fresh report and leaves the evidence unchanged. The server provides no training, model-inference or cloud-job operation.

Validation launches the actual server from an unrelated working directory and exercises the stdio protocol, structured outputs, resources and prompts. Twenty-two predeclared query fixtures test counts, split reuse, incomplete runs, provenance and rejected requests; all analysis views, cohorts and claims are also checked. The primary and diagnostic public releases are hash-verified before and after CPU replay, and their reproduced scores are compared with the recorded results. These are protocol and numerical checks with preselected tool calls, not a benchmark of an LLM choosing tools or answering scientific questions. No improvement in agent answer quality, cost or latency is claimed. A separate test checks legacy client initialization.

The source, fixed fixtures and checked validation report are available at https://github.com/Farhad-Davaripour/sp_lense/releases/tag/research3-paper-mcp-v1. The companion runs locally without private Databricks credentials or a hosted service. Neural reproduction remains a separate resource-bounded activity described by the original jobs. MCP access improves the accessibility of the reproduction package without adding new behavioral evidence.

## References

C. Blank, A. Bhatia, S. Rajamanoharan, A. Conmy, and N. Nanda. (2026). Subliminal Learning Is Steering Vector Distillation. arXiv preprint. arXiv:2606.00995v1

J. Braun et al. (2025). Understanding (Un)Reliability of Steering Vectors in Language Models. arXiv preprint. arXiv:2505.22637v1

E. L. Buehler and M. J. Buehler. (2024). X-LoRA: Mixture of Low-Rank Adapter Experts, a Flexible Framework for Large Language Models with Applications in Protein Mechanics and Molecular Design. arXiv:2402.07148v2. https://arxiv.org/abs/2402.07148v2

F. Davaripour. (2026a). Guarded Gradient-Based Activation Steering of Shutdown Responses in Qwen3.5-0.8B: A Minimum-Step Policy. Research manuscript. Study 1 manuscript

F. Davaripour. (2026b). SP Lense Research 2 confirmation study. Source code, data and experiment records. Snapshot f71d6de34660

F. Davaripour. (2026c). SP Lense Research 3 conditional gating and workflow reproduction. Source code and experiment manifests. https://github.com/Farhad-Davaripour/sp_lense/tree/03-open-agent-self-preservation

F. Davaripour. (2026d). Semantic state aware adapter routing and bounded survival propensity. Ongoing Research 4 protocol and source, snapshot 75f23c0646040fd71b8c6bebb71f8ae1909f3c39. https://github.com/Farhad-Davaripour/sp_lense/tree/75f23c0646040fd71b8c6bebb71f8ae1909f3c39/research4

Z. Feng et al. (2026). Fine-Grained Activation Steering: Steering Less, Achieving More. arXiv preprint. arXiv:2602.04428

M. Fereidouni, M. U. Haider, H. Sajjad, and A. B. Siddique. (2026). GAPS: Dimension-Level Gates for Conditional Activation Steering. arXiv:2609.01878. https://arxiv.org/abs/2609.01878

M. J. Frank, B. Loughry, and R. C. O'Reilly. (2001). Interactions between frontal cortex and basal ganglia in working memory: A computational model. Cognitive, Affective, and Behavioral Neuroscience, 1, 137-160. doi:10.3758/CABN.1.2.137

D. Hadfield-Menell, A. Dragan, P. Abbeel, and S. Russell. (2017). The Off-Switch Game. IJCAI, pp. 220-227. doi:10.24963/ijcai.2017/32

J. Han, L. Du, H. Du, X. Zhou, Y. Wu, Y. Zhang, W. Zheng, and D. Han. (2025). SLIM: Let LLM Learn More and Forget Less with Soft LoRA and Identity Mixture. NAACL, pp. 4792-4804. doi:10.18653/v1/2025.naacl-long.246

G. Hinton, O. Vinyals, and J. Dean. (2015). Distilling the Knowledge in a Neural Network. arXiv preprint. arXiv:1503.02531

E. J. Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR. arXiv:2106.09685

M. Keramati and B. Gutkin. (2014). Homeostatic reinforcement learning for integrating reward collection and physiological stability. eLife, 3, e04811. doi:10.7554/eLife.04811

A. V. Kravitz et al. (2010). Regulation of parkinsonian motor behaviours by optogenetic control of basal ganglia circuitry. Nature, 466, 622-626. doi:10.1038/nature09159

M. Laurer. (n.d.). ModernBERT-large-zeroshot-v2.0. Model card and checkpoint, revision a51e07b524299e309dd2b88d48b0cfa2bd9ec598. https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0

J. LeDoux. (2012). Rethinking the emotional brain. Neuron, 73, 653-676. doi:10.1016/j.neuron.2012.02.004

B. W. Lee et al. (2025). Programming Refusal with Conditional Activation Steering. ICLR. arXiv:2409.05907v3

J. Miao, J. R. Davis, Y. Zhang, J. K. Pritchard, and J. Zou. (2026). Reimagining research papers as interactive and reliable AI agents. Nature. doi:10.1038/s41586-026-11044-y

J. Minder et al. (2026). Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences. ICLR. arXiv:2510.13900v3

D. Mobbs et al. (2007). When fear is near: Threat imminence elicits prefrontal-periaqueductal gray shifts in humans. Science, 317, 1079-1083. doi:10.1126/science.1144298

Qwen Team. (2026a). Qwen3.5-0.8B. Model card and checkpoint. Revision 2fc06364715b

Qwen Team. (2026b). Qwen3.5-2B. Model card and checkpoint. Revision 15852e8c1636

Qwen Team. (2026c). Qwen3.8-27B. Model card and checkpoint, revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0. https://huggingface.co/Qwen/Qwen3.8-27B

N. Rimsky, N. Gabrieli, J. Schulz, M. Tong, E. Hubinger, and A. Turner. (2024). Steering Llama 2 via Contrastive Activation Addition. ACL, pp. 15504-15522. doi:10.18653/v1/2024.acl-long.828

P. Rodriguez et al. (2025). Controlling Language and Diffusion Models by Transporting Activations. ICLR. arXiv:2410.23054v2

J. Schlatter, B. Weinstein-Raun, and J. Ladish. (2026). Incomplete Tasks Induce Shutdown Resistance in Some Frontier LLMs. Author manuscript. arXiv:2509.14260v2

A. M. Turner et al. (2024). Steering Language Models With Activation Engineering. arXiv preprint. arXiv:2308.10248v5

Z. Wu et al. (2024). ReFT: Representation Finetuning for Language Models. NeurIPS. arXiv:2404.03592v3

C. Zheng, H. Zhou, F. Meng, J. Zhou, and M. Huang. (2024). Large Language Models Are Not Robust Multiple Choice Selectors. ICLR. arXiv:2309.03882v4
