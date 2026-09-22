# CA3-state goal follow-up — 22 September 2026

Status: follow-up fixes and deferred refinements for the deployed predictive active-goal design.

## Immediate fixes

1. **Contextual HER success semantics.** DG ID is only the landmark slot/address. A HER goal selected from future raw CA3 state S_g must be achieved with the same contextual recognition semantics as online goals. Keep the slot as a cheap candidate lookup, but do not let slot equality alone define success:

   hit(S_t,S_g) = [j_t = j_g] [sim(sigma(S_t), sigma(S_g)) >= tau].

2. **Restore CA3-readout anti-collapse regularization.** The implementation currently has the prediction loss but omitted the planned variance/covariance terms. Restore

   L_state = L_pred + lambda_var L_var + lambda_cov L_cov,

   initially lambda_var = 0.1 and lambda_cov = 0.01. Keep state/action shuffle deltas as diagnostics of whether the predictor actually uses z and actions.

3. **Task-general transfer mode.** Preserve the historical DG/policy transfer scopes, but add a mode that transfers the task-general learned system: DG representation, W, innovation predictor, contextual anchors/graph, universal worker/controller and their normalization/state. Leave task-specific external-reward bindings/heads fresh. Keep the action interface matched unless action remapping is the explicit experiment.

## Future refinement: counterfactual probe bank

The repeated-single-action probes are acceptable for V1. If contextual signatures are unstable or insensitive, replace them with a fixed shared bank of behavior-supported replay action prefixes at several horizons. Every state must still be queried with the same probe bank; the purpose is only to reduce out-of-distribution counterfactual sequences.

## Future refinement: recognition threshold calibration

Keep positive-only calibration as V1 because within-occurrence pairs are trusted positives, whereas apparent negatives can be false negatives under route invariance or redundant DG codes.

Add background/negative diagnostics without negative training:
- cross-anchor pairs;
- temporally separated occurrences of the same DG slot;
- different-DG occurrences;
- offline coordinate-verified different physical locations.

Report positive/background similarity distributions, overlap or ROC-style diagnostics, and false-positive estimates where privileged offline labels are available. Add explicit negative training only if these diagnostics show that positive-only calibration cannot separate aliases.

## Future refinement: anchor update semantics

Separate same-identity refinement from semantic reset. Confirmed same-landmark observations may refine the anchor while preserving graph edges and generation. Clear incident graph evidence only when there is evidence that the slot identity itself changed.

Use a clean two-condition comparison:

1. **FIXED_ANCHOR.** Keep the first qualified raw CA3 anchor A_j unchanged after activation. This is the stability control.
2. **EMA_SIGNATURE_ANCHOR.** Keep A_j as one actually observed raw CA3 state, but maintain an exponential-moving-average predictive-signature prototype

   mu_j <- (1 - alpha) mu_j + alpha sigma(S_t)

   over confirmed same-landmark occurrences. Do not average raw CA3 states. A confirmed candidate S_t may replace A_j only when its predictive signature is closer to mu_j than the incumbent's by a small margin or sustained criterion. Such refinement preserves graph edges and anchor generation because semantic identity has not changed.

This gives gradual prototype refinement without ever commanding a synthetic/nonexistent CA3 state. A true semantic reset remains a separate rare operation that increments generation and clears incident graph evidence.

The intended semantics remain one landmark per DG slot; this is not a contextual-clone proposal.

## Future refinement: simultaneous DG activations

Preferred rule if it remains a small local change:

1. collect all currently active and selectable DG slots;
2. evaluate contextual recognition for each active slot;
3. accept the landmark if exactly one active slot passes its contextual threshold;
4. if zero or multiple slots pass, mark the event ambiguous and do not update landmark/graph evidence.

This is less restrictive than raw exclusivity: several DG units may be active, but one unambiguous contextual landmark can still be recognized.

Implementation priority:
- **preferred:** unique contextual match, because the existing action-probe signature and recognition threshold can be reused and the current contextual_activity path already computes the needed score;
- **fallback if the patch ceases to be local/simple:** strongest/dominant DG activation, matching the historical visit_direct convention;
- **later only if abstention is excessive:** best contextual score with an explicit margin over the runner-up.

Keep this choice configurable so strongest-DG and unique-contextual-match can be compared directly.

Historical context: legacy visit_direct used dominant/argmax recognition. Later frontier_direct/frontier_waypoint and the frozen DDQN/HER parent used exclusive-positive landmark recognition. Treat this as a precision-versus-coverage ablation rather than assuming either rule is universally best.