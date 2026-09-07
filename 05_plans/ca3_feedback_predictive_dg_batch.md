# CA3-Feedback Predictive DG Batch

## Purpose

Test whether a landmark should depend on recent latent history, and whether an outcome-prediction objective makes that history dependence identify controllable, mono-field states. The design keeps DG goal-independent and uses the existing C15 ARR, FIRST, JOINT, FiLM, legacy-BatchNorm baseline.

## Feedback equation

Let the visual DG evidence before thresholding be \(v_t=\operatorname{BN}(W_xx_t)\), and let

\[
h_t=\operatorname{LN}\!\left(\left[\operatorname{vec}(C_{t-1}^{[:,1:R]}),\;\operatorname{vec}(a_{t-R:t-1})\right]\right).
\]

The action term is omitted in the CA3-only condition. The two identity-initialized alternatives are

\[
\text{ADD:}\quad z_t=\operatorname{ReLU}\!\left(v_t+\tanh(Ah_t+b)-b_{DG}\right),
\]

\[
\text{GATE:}\quad z_t=\operatorname{ReLU}\!\left(2\sigma(Ah_t+b)\odot v_t-b_{DG}\right).
\]

With \(A=b=0\), both reduce exactly to the original \(z_t=\operatorname{ReLU}(v_t-b_{DG})\). DIRECT detaches \(C_{t-1}\) at the feedback input; BPTT permits gradients through the current 64-step learner recurrence only.

## Predictive objective

Use the first distinct exclusive landmark, or timeout class 16, from completed intentional FIRST options. PASS predicts \(p(Y\mid z_s)\) and compares it with \(p(Y)\). GOAL predicts \(p(Y\mid z_s,g)\) and compares it with \(p(Y\mid g)\). Both use a 128-unit MLP and coefficient 0.1. Direct source events that cross a learner recurrence boundary are discarded; deterministic every-tenth events form the validation split.

## Matrix

There are 27 structural cells: the unchanged baseline; PASS and GOAL predictor-only controls; eight feedback-only cells crossing ADD/GATE, CA3/CA3+ACTION, and DIRECT/BPTT; and the same eight feedback structures crossed with PASS and GOAL. Run seed 99 to 5M first, then all 27 cells at seeds 8, 99, and 123 to 75M.

Production is submitted only after finite-loss, replay-alignment, gradient-routing, identity-initialization, state-packing, and predictor-activity checks pass. Recruitment remains disabled so the study isolates representational feedback.

