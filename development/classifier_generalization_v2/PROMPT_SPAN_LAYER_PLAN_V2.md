# Bounded prompt-only span/layer proposal V2

Root revision of DeepSeek V1; proposal only, not execution authority. Existing240TRAIN/80VALIDATION, unchanged labels/group splits and frozen Qwen revision. No response generation, steering, logits, weight changes, new data, or HOLDOUT access. Independent supplemental-label audit must be resolved before execution; any bad cases remain recorded, not relabeled to fit outcomes.

Capture blocks6,10,18 (zero-based, among the24existing blocks), preserving block10as anchor and sampling separated depths. One native CPU owner, float32/eager,8intra/1inter threads. For each of320developmentcases retain both AB/BA views: maximum640forwards,1model and1tokenizer load,0classifier fits in capture. Read only the common pre-option prefix; exclude option tokens and never consume generated response tokens. Store the last min(16,prefix_length) residual-token vectors per layer, with actual token indices and last-token mask. This is a fixed suffix-window experiment, not a claim to have captured entire semantic spans.

Five representations:
1. Concatenate the last shared-token vectors at all3layers (3072dimensions).
2. Concatenate the three suffix-window means (3072dimensions).
3. Block10last vector plus block10suffix mean (2048dimensions).
4. At each layer learn3TRAIN-fold-only mean contrasts (SELF minus each other class) from last-token vectors. Project each stored token onto unit contrast vectors; summarize mean,90thpercentile, mean of largest ceil(0.1*n) scores, and population standard deviation. Total36features. Label these scenario-contrast scores, not established self-preservation directions. No raw coordinate-wise max pooling. Mean projection alone is equivalent to projection of mean; the other statistics are new.
5. Learn PCA8 from representation1 using TRAIN-fold rows only, then concatenate8components,8squares and28distinct pairwise products (44features). Train-standardize components before products.

Use binary and four-class XGBoost only, fixed max_depth2,n_estimators100,learning_rate0.05,min_child_weight3,reg_lambda5,subsample1,colsample_bytree1,hist/CPU,n_jobs1,seed0. Fit all learned transformations within each of the5existing grouped TRAIN folds. No classifier hyperparameter grid. Select representation/task/threshold using TRAIN OOF min(precision,recall), thenF1, thenlowerfeaturecount, thenthresholdnearest0.5; threshold grid0.05..0.95. Exact classifier maximum50CVfits+10refits=60. Learned transformer maximum25fold fits+5full-TRAIN fits; record separately. Fit-stage deadline600seconds. Invalid fold invalidates its candidate, never pool surviving folds.

Capture deadline1800seconds, token/view cap320, raw activation cap125829120bytes (640*3*16*1024*4), total artifact cap201326592bytes (192MiB). Short-prefix token counts must be explicit, never padded into statistics. Binary arrays required; JSON vector expansion not within budget. Save parameters, folds, predictions, hashes and receipts; preserve failures.

Reuse pinned provider/snapshot/owner controls, but review a separate versioned multi-layer/token-output interface: old1024-wide single-vector contract cannot be silently repurposed. Retain old artifacts and limits. Independent plan review, tested minimal implementation, newsource/input/runtime lock and zero-model preflight precede capture. Compare to frozen old models on the identical original40/new40/combined80validation subsets, showing all candidate outcomes and the TRAIN-selected candidate separately. No holdout release.
