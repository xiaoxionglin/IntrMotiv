# Three recent architecture batches: synthesis (24 September 2026)

## Reading order and evidence boundary

1. [Easy landmark maze: completed 2M qualification](easy_landmark_maze_qualification_analysis_20260924.md) asks whether visible, fixed cues change localization, exploration, and control across three existing architectures. Its 12 production runs have since started and remain in progress.
2. [CA3 predictive active goals: 75M interim analysis](ca3_predictive_active_goals_interim_analysis_20260924.md) compares seven ways to turn recent CA3 history into worker state, goals, and context recognition. All 21 runs have a 75M online spatial snapshot; none has completed the declared 300M horizon.
3. [CA3 state-goal follow-up: 25M interim analysis](ca3_state_goal_followup_interim_analysis_20260924.md) holds the predictive state-goal design fixed and crosses two anchor-update rules with two candidate-recognition rules. All 12 runs have a 25M snapshot; ten have a 75M snapshot.

The first report is a qualification result; the other two are synchronized interim comparisons. Their environments differ: the landmark batch uses a fixed 74-cell maze with visible/neutral cue conditions, whereas both CA3 batches use the established reward-free open field. The three tables must not be pooled into an architecture ranking.

The landmark training audit and frozen evaluator passed, but separate exact learner-reload certificates were not found in the qualification analysis tree. The CA3 online scalar histories are incomplete in the collector's discovered event-file path after recovery, so their matched findings here come from validated spatial snapshots. These gaps are called out in the individual reports.

## How the components fit together

The common visual trunk is fixed ImageNet ResNet-18 through layer 2. A learned DG projection with evolving BatchNorm statistics supplies sparse landmark activity; CA3 retains recent landmark history. The controller receives a selected target and learns action values with stored-state DDQN and hindsight experience replay (HER) in the waypoint/CA3 line. The landmark screen also includes two earlier PPO-based graph/goal designs. Pose and cue identity are privileged telemetry only, not policy inputs.

| Question | Controlled comparison | Current observation | What remains unresolved |
|---|---|---|---|
| Do visible cues improve exploration? | Rich versus neutral rendering at the same reserved sites, paired within seed 99 and architecture | Frozen two-episode coverage AUC improves for SCR and Waypoint, declines for DGP; the landmark training window gives a different, noisier pattern | Three-seed production and longer held-out episodes |
| Does a predictive CA3 state improve goal-directed behavior? | Seven predictive/goal/context architectures at matched 75M and three seeds | Map and graph properties vary; every cell has mean grounded controllability 0 at 75M | Matched online behavior and controlled offline command outcomes through later training |
| Which contextual-goal rule helps? | Fixed/EMA anchors crossed with dominant/unique contextual candidates at matched 25M and three seeds | Unique candidates show more mono-field units but nearly empty reliable graphs; fixed anchors have lower map overlap than EMA in this early window | Later matched horizons, actual goal recognition, and controlled success |

## Practical interpretation

Localization, exploration, and control are separate outcomes. A low active-map cosine means DG maps overlap less on visited bins; it does not show that each unit has one place field. Many reliable graph edges or high reachability describe an internal transition model; they do not show that issuing a command changes the physical destination. Accessible coverage measures where a policy goes, but not whether a chosen goal caused it to go there. The reports therefore retain mono-field fraction, silent units, graph reachability, grounded control, and coverage alongside one another.

The main scientific tension is now visible: extra state-goal machinery can alter representation and graph structure without yielding demonstrated commanded control. The follow-up's unique rule appears to trade early graph density for more localized units; that is a mechanism hypothesis, not yet a navigation win. The landmark result shows that visually salient cues can help one architecture while hurting another, making the architecture-by-environment interaction a production question.

## Shared next analysis point

Update these reports at the next complete synchronized checkpoint. For the predictive batch, use the canonical 75M, then 150M and 300M schedules only when every paired cell is present. For the follow-up, use 25M now and 75M once both lagging fixed/seed-8 runs arrive. For the landmark production, analyze the rich three-seed cells and the paired seed-99 neutral controls at their declared shared milestones. Add manifest-driven frozen place fields and matched-command interventions before making a control claim.

## Reusable workflow lesson

The StudySpecs and `collect-spatial` provided exact identities, synchronized targets, and comparable definitions with little new analysis code. The first remote attempt used the system Python without NumPy; the working runtime was `/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python`. The main NEMO2 checkout was at workflow 1.10.1 and could not load the follow-up's 1.11.0 StudySpec; the already verified easy-landmark 1.12.0 checkout could. The remote follow-up StudySpec was absent from that checkout and was staged as a lightweight analysis input in the allocated workspace. Next time, resolve both the runtime Python and workflow version before launching the collector. The existing `infra.md` workflow-version and storage entries cover these recurring concerns; no duplicate tracker entry is needed.
