# RB-HRNet / LAF experiment status

| Experiment | Backbone | Neck | Status |
|---|---|---|---|
| E0 | Existing PResNet-18 | Existing FPN/PAN | Historical metrics supplied; original run artifacts not audited |
| E1 | Plain HRNetV2-W18 | Existing FPN/PAN | New Ultralytics control YAML; historical weights not found in this tree |
| E2 | RB-HRNet-A | Existing FPN/PAN | Implemented; not trained |
| E3 | RB-HRNet-B | Existing FPN/PAN | Implemented; not trained |
| E4 | Plain HRNetV2-W18 | LAF-0 | Implemented; not trained |
| E5 | Plain HRNetV2-W18 | LAF-1 | Implemented; not trained |
| E6 | RB-HRNet-A | LAF-1 | Implemented; defer training until E2–E5 validation |

For each completed experiment, record overall AP50, AP75 and AP50:95;
Tiny (≤8 px), Small (8–16], Medium (16–32], Large (>32 px) AP and recall;
parameters, measured MAC scope, latency and GPU memory. Flag Medium or Large
AP50:95 reductions greater than 0.003. Do not select architecture by repeated
test-set inspection.
