# Grasping Rollout Context

This context is intentionally separate from the embodiment's base system
documents. Use it only for a controlled prompt-ablation rollout.

- After contact, do not immediately carry laterally.
- Close the gripper, lift 3--5 cm, and verify in both camera views that the
  object follows the gripper before carrying it.
- If two consecutive actions are delta-clamped, split the motion into smaller
  segments and re-check the observation.
- If the wrist orientation drifts away from vertical, restore orientation in a
  separate action before continuing.
- For bowls, prefer gripping the side wall with one pad inside and one outside;
  do not rely on pinching the thin rim.
- Do not spend the final calls on an unverified close-and-lift gamble.
