---
status: unresolved
next_action: Convert into a formal note on global CA3-state reward credit-assignment failure modes.
---

# Unresolved: Global CA3-State Credit Assignment

master comment:
we tried some more global versions where the entire CA3 states were used. But due to the way how actor-critic algorithm works, the single DG activation way seems to be better. My intuition is, the past distances are not changed by future actions, if using the entire CA3 state, you will be rewarding all following actions with one good past action. That would create a lot of unwanted reinforcement signal, leading to strengthening the chosen actions without a reason. Please think about it and reason with me.
