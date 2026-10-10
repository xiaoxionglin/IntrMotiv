Can spatial representation emerge without any spatial supervision and how?
We designed a system with temporal push pull mechanism based on an architecture inspired by hippocampal DG CA3 physiology, without external spatial reward, spatial supervision or explicit spatial objectives.
- spatial information increases with training under this loss function
- representation is shaped by behavior, which is inherently spatial (random policy gives worse spatial information)
- temporal push for DG learning increases spatial spread of DG place fields (also rate map cosine; spatial info?)
- temporal pull reward for policy increases spatial coverage of trajectories
- hierarchical RL improves trajectory (need more thought if this should go to the beginning)