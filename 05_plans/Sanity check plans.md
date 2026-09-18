1. **Can the existing DG architecture learn good fields with explicit neighborhood supervision?**  
    Compare the existing objective, neighborhood supervision alone, and supervision plus **temporal versus physical local repulsion**. Train all units through pre-threshold scores while preserving sparse DG→CA3 activity. Measure compactness, disconnected fields, recall, false positives, silence, and held-out generalization.  
    **Already implemented and launched**, with preliminary results in the [DG experiment report](/home/xiaoxiong/Desktop/Projects/IntrMotiv/06_experiments/dg_neighborhood_g500_20260914.md). The implementation evolved from the original fixed-center oracle proposal.
    
2. **Did PPO gradients create the useful DG landmarks in the successful fixed-reward task?**  
    Start with PPO→DG **on versus off**. Then separate **actor-only, critic-only, both, and neither**. Check navigation performance and whether fields concentrate near useful turns, bottlenecks, or trajectory starting points.
    
3. **Does temporal credit through CA3 shape DG?**  
    Preserve the same forward memory but block gradients into **past DG activations**, retaining current-time gradients. Compare against full temporal backpropagation. This distinguishes CA3’s contribution to learning landmarks from its contribution as memory.
    
4. **Does our architecture work better when navigation consists of corridor-like segments?**  
    Systematically compare **corridor → Y-maze → small/deep tree → tree with loops → open field**. Vary branching, corridor length, and loops. Initially keep CA3 goal-independent and condition the decoder on the goal. This was your main geometry hypothesis.
    
5. **Can structured memory compensate for sparse visual landmarks?**  
    Vary **visual-anchor density × distance between anchors**, comparing feedforward, LSTM, current CA3, and a movement-modulated CA3 clock. The prediction was that CA3’s advantage should grow as reliable sensory anchors become farther apart.
    
6. **Would limited action dependence improve CA3?**  
    Proposed progression: current fixed clock → pause/advance → speed modulation → action/context selecting parallel sequence chains → branching at landmark events. These were exploratory extensions, not an agreed large training matrix. We also clarified that changing clock speed gives trajectory odometry, not full 2-D path integration.
    

Two additional diagnostics are recorded in the project plans:

- **ARR on versus off**, with the same JOINT controller and maintenance terms, to isolate whether the encoder objective is harmful. [Objective scrutiny](/home/xiaoxiong/Desktop/Projects/IntrMotiv/05_plans/encoder_objective_scrutiny_20260914.md)
- **Four perfect oracle fields versus learned goal identities**, to ask whether accurate landmarks rescue control. Evaluate physical arrivals against shuffled-command execution, not just online hit rate. This is distinct from training visual DG with oracle labels. [Oracle control plan](/home/xiaoxiong/Desktop/Projects/IntrMotiv/05_plans/oracle_dg_place_fields_20260914.md)