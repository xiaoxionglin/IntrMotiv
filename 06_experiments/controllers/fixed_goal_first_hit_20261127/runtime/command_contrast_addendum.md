# Command contrast: distinguish navigation from goal-dependent stopping

Registered at **2026-11-27 09:27:30 UTC**, before main efficacy outcomes, following a logical protocol audit. The main replay budget, learned objective, deployed policy, starts, commands, horizons and recorded trajectories are unchanged. This addendum supersedes interpreting stopped-task command lift as evidence of command-dependent navigation. It is recorded before the main launch bundle is regenerated.

## Why the original lift is confounded

Own-goal stopping is part of the requested task and should remain in executed evaluation. However, comparing arrival at $g$ under command $g$ against arrival at $g$ under command $c$ mixes action selection with stopping. Consider a command-independent path $s\to c\to g$ within the deadline. Command $g$ reaches $g$; command $c$ stops at $c$. Their stopped-task arrival difference is one even though every action taken before stopping is identical.

The original stopped-task lift remains a secondary description of the complete goal-and-stop interface. It does not establish selective navigation. Exact-start fitted-versus-source own-goal success remains a useful task comparison because both policies receive the same command and stopping rule.

## Primary command-selectivity contrast

Fix a physical/recurrent start history $x$, deadline $H$, frozen actor $p$, and distinct goals $g,c$ that are both absent from the initial winner observation. Let $\tau_j^{p,u}$ be the first decision-time winner event for goal $j$ under command $u$, with no event represented by infinity. Use the same fixed, goal-independent processed-DG winner predicate for both commands. Define

$$
Z_{g,c}^{p,u}(x)=\mathbf{1}\{\tau_g^{p,u}\le H,\ \tau_g^{p,u}<\tau_c^{p,u}\},\qquad u\in\{g,c\},
$$

and

$$
\Delta_{g,c}^{p}(x)=\mathbb{E}[Z_{g,c}^{p,g}(x)]-\mathbb{E}[Z_{g,c}^{p,c}(x)].
$$

Only $g$ and $c$ compete in this diagnostic. Other landmark passages are allowed. This does not change the training objective into failure on the first different landmark, and it does not require new rollouts. For either command, its own stop cannot precede the first event in $\{g,c\}$. Existing cross-target `hit_time` rows therefore identify the event up to physical-terminal censoring. An observation that arrives exactly at $H$ counts; events between sampled decisions are not inferred.

For a deterministic command-blind actor, identical engine/prefix states and common environment dynamics give the same trajectory until the first member of $\{g,c\}$. Thus $Z_{g,c}^{p,g}=Z_{g,c}^{p,c}$ and the contrast is zero, despite different later own-stop times. With stochastic policies or dynamics, the analogous null is equality of these event distributions; an individual sampled difference need not vanish. Command-dependent observation/dynamics or unequal start/deadline distributions would invalidate that interpretation.

## Classification, bounds and interpretation

For each executed command, compare the recorded first times for $g$ and $c$:

- First $g$ establishes $Z=1$; first $c$ establishes $Z=0$. Either remains established if a later physical terminal is unobserved.
- No event through a fully observed deadline establishes $Z=0$. A certified physical terminal with neither goal also establishes zero; the current evaluator does not observe its terminal DG.
- An unobserved physical terminal before either event, including an unobserved final decision at $H$, leaves $Z\in[0,1]$. Missing starts, unissued commands or missing necessary target rows are also unknown. Do not turn them into observed failures.
- An initially satisfied $g$ or $c$ is outside this contrast's eligibility set, but remains in task-success and capacity-denominator reports. Equal first times for distinct winner IDs contradict the one-winner predicate and require an artifact-consistency failure, not arbitrary tie-breaking.

If the two command outcomes have bounds $[L_g,U_g]$ and $[L_c,U_c]$, the paired contrast has the conservative identification interval

$$
\Delta_{g,c}^{p}(x)\in[L_g-U_c,\ U_g-L_c].
$$

Average endpoints using the same prespecified nonnegative weights over declared histories/goals/competitors. Preserve missing and censored cases; never condition the primary contrast on one target being reached, both trials being complete, or an observed favorable route. Missing target capacity is unassessed rather than empirically unreachable. Identification bounds quantify unresolved outcomes; they are not confidence intervals. Shared histories induce dependence across goals and competitors, so uncertainty estimates must preserve the declared source-history clustering and remain conditional on this single donor.

A positive competing-target contrast establishes command-dependent ordering under the stated coupling assumptions. Zero does not by itself refute goal reaching: a corridor may require passing $c$ before $g$ under every successful policy, or the available actions/information may prevent changing their order. Do not require every ordered pair to have positive contrast or select favorable pairs after seeing outcomes. Any routing-choice strata must be defined independently of efficacy outcomes. Report the full matrix and aggregate selectivity together with complete per-goal arrival evidence. This separates a useful goal-dependent stopping interface from evidence that commands choose different navigation behavior.
