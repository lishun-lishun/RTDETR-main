# Targeted experiment analysis

Populate this report from `tools/analyze_r18_targeted.py` after each run.  The
script reads only saved evaluation artifacts and records the declared split,
so a test-set result is never labelled as validation data.  Compare mAP50,
mAP50-95, precision, recall, size-group AP and localization CSVs under a
common seed and protocol.  Missing checkpoints or metrics are reported as
missing rather than imputed.
