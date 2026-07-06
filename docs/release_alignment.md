# Release Alignment Notes

This note records the final choices made before publication by comparing the
clean release tree with the reproduction worktree that was used for the
Section 3.1 rerun.

## Chosen Versions

1. `verl/verl/trainer/ppo/ray_trainer.py`: use the reproduced version for the
   SwanLab plotting guard. Plotting now runs only when `swanlab` is present in
   `trainer.logger`, so the default console-only run does not require SwanLab.
2. `verl/verl/workers/actor/dp_actor.py`: use the reproduced version for the
   Direct-OPD reward definition. The release does not include the experimental
   `DELTA_ANSWER_PROTECT_*` answer-tail protection path.
3. `verl/verl/workers/actor/dp_actor.py`: keep the release version for top-k
   log-prob computation. The release keeps the existing `logsumexp`/gather
   implementation.
4. `scripts/train_justrl_qwen.sh` and public docs: keep the release layout and
   defaults. Local absolute paths and one-off rerun overrides from the
   reproduction worktree are intentionally not published.

## Remaining Intentional Differences From The Rerun Worktree

- The release has one public entrypoint, `scripts/train_justrl_qwen.sh`; the
  rerun worktree kept older experiment-specific scripts.
- The release uses portable defaults under `models/`, `datasets/`,
  `checkpoints/`, and `logs/`; the rerun worktree used local absolute paths
  under the shared home directory.
- The release documents and packages small evaluation parquets under
  `datasets/eval/`; the rerun worktree used local OPD-test evaluation files.
- The release keeps the compact public README/docs instead of the older
  reproduction README from the rerun worktree.
- Generated checkpoints, logs, plots, and Hydra outputs from the rerun worktree
  are not part of the publishable repository.
