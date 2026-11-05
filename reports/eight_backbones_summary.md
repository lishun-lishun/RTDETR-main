# Eight-backbone RT-DETR summary

This table is a result template. Run `tools/summarize_backbones.py` after the
validation/test and benchmark artifacts exist. Ranking is based on AP50:95 and
AP75 together with Tiny/Small AP, Medium/Large AP, FPS, latency and GPU memory.

| Model | Params | GFLOPs | AP50 | AP75 | AP50:95 | Tiny AP | Small AP | Medium AP | Large AP | FPS | Latency | GPU Memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hgnetv2_b2 | | | | | | | | | | | | |
| convnextv2_pico | | | | | | | | | | | | |
| fasternet_t1 | | | | | | | | | | | | |
| repvit_m1_1 | | | | | | | | | | | | |
| efficientvit_b1 | | | | | | | | | | | | |
| pvt_v2_b1 | | | | | | | | | | | | |
| lsknet_t | | | | | | | | | | | | |
| pkinetv2_t | | | | | | | | | | | | |

No result is pre-filled before the same validation protocol, test split and
benchmark settings have been run for all candidates.
