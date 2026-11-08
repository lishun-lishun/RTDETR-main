# RT-DETR-R18 targeted variants R09--R16

This delivery adds eight independent backbone YAMLs for DUT-Anti-UAV.  Each
variant keeps the original RT-DETR-R18 neck, decoder and loss contract.  R09
and R10 replace the Stage3 downsampling path; R11--R13 and R16 refine P3 once;
R14 replaces only Stage4 block two's branch; R15 inserts direct P3/P4 semantic
feedback, feeding refined P3 to the neck while original P4 continues to Stage5.

The implementation is opt-in.  `rtdetr-r18.yaml` is unchanged and no training
job starts during import, testing, or launcher dry-runs.  Use
`tools/train_r18_targeted.py --model <name> --device <id> --execute` after the
dataset and protocol audits pass.  Evaluation takes an explicit `--split`
(`test` is the default), so validation and test metrics cannot be confused.

The operators expose lightweight diagnostics (protect/contrast/saliency,
branch weights, and SDF gate) for post-run analysis.  No ground-truth tensor,
hard threshold, top-k selection, or auxiliary loss is introduced.
