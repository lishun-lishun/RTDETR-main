# RT-DETR-R18 SA-Rep / SP-ASFF implementation audit

## Source audit

- `ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml` has eight backbone layers and a CCFM neck ending at layers 19 (F3), 22 (F4), and 25 (F5). Decoder layer 26 receives `[19,22,25]`, 256 channels each, and 300 queries.
- `ultralytics/nn/modules/block.py` defines `BasicBlock` as `branch2a Conv+BN+ReLU -> branch2b Conv+BN -> shortcut add -> ReLU`. `Blocks` constructs two blocks per stage; Stage3's second block is `blocks[1].branch2b`, with stride 1 and identity shortcut.
- `ultralytics/nn/tasks.py` resolves `Blocks` specially; the new `SARepBlocks` and `SPASFF` need their own channel-argument parser cases.
- `ultralytics/engine/trainer.py` uses `nw = self.args.warmup_epochs` directly, so the field means **micro-batch iterations**, despite its name. `train_dut_anti_uav.py` multiplies 2000 by accumulation steps. With checked-in micro-batch 2 and effective batch 8, the trainer receives 8000 micro-batch warmup iterations, or about 2000 optimizer updates. The schedule has not been changed.
- No completed original R18 `args.yaml`, `results.csv`, or checkpoint was found in this local tree. Therefore historical AP values and exact baseline training configuration have not been independently verified here.

## Added modules

`ultralytics/nn/modules/sarep_spasff.py` contains:

- `SARepConv`: Conv+BN branches with 3x3, 1x3, and 3x1 kernels. The deployment conversion folds each branch's BN running statistics and affine parameters, pads asymmetric kernels to 3x3, then sums weights and biases. Grouped convolution and device/dtype are preserved. `switch_to_deploy` removes all three training branches.
- `SARepBlocks`: constructs the unchanged original `Blocks` stage, then replaces only selected `branch2b` modules. Default E1 changes Stage3 block index 1. The first block's downsample and both residual paths remain original.
- `SPASFF`: projects F3/F4/F5 to 64 channels, resizes F4/F5 with nearest neighbor, predicts three per-pixel softmax weights, and returns only F3 plus a bounded, zero-initialized difference residual. Its `direct` and `no_difference` options are ablations. The original decoder F4/F5 references remain at layers 22/25.

The YAML files are derived from the original R18 graph without modifying its CCFM internals, BasicBlock class, loss, decoder, P2 head, or query count. The optional ordinary ASFF control (`asff_direct`) replaces only decoder F3 using the same three projected scales and the same fusion location; F4/F5 remain untouched. It thus tests direct replacement versus protected residual at a comparable fusion location, while the zero-gain difference and no-difference pair isolates the differential term.

## Training and server validation

From `RTDETR-main`, on one physical GPU:

```bash
python -u tools/train_r18_sarep_spasff.py --model sarep --device 0 --baseline-args /path/to/original-r18/args.yaml
python -u tools/train_r18_sarep_spasff.py --model spasff --device 0 --baseline-args /path/to/original-r18/args.yaml
```

Use `--dry-run` to inspect effective settings without importing PyTorch. Without `--baseline-args`, the script uses this repository's checked-in `anti_detr_default.yaml` and prints that historical protocol parity is unverified. With `--baseline-args`, the script checks effective batch 8 and the stated optimizer protocol, then preserves all baseline settings present in Ultralytics defaults, including augmentations, micro-batch, gradient accumulation, and warmup micro-steps. It never loads baseline best.pt into a formal run. The trainer itself writes `args.yaml`, per-epoch `results.csv` (including actual optimizer learning rates), and checkpoints. The launcher writes `training_config.json`, `model_summary.json`, `checkpoint_audit.json`, and `execution_log.txt` in a new experiment directory. Existing output is refused. Resume uses `--resume /path/to/last.pt`.

Run the complete server suite before training:

```bash
python -m unittest discover -s tests -p test_sarep_spasff.py -v
```

The launcher also runs that suite and checks decoder input shapes before a new run. The local CPU environment passed standalone SA-Rep numeric equivalence, residual placement, gradients, SP-ASFF identity, weight normalization, two-step gradient, batch-size/shape, and YAML tests. Full Ultralytics graph and checkpoint tests were skipped locally because `cv2` is absent; CUDA AMP was skipped because this Windows environment has no CUDA. A30 results must be checked on the server. No AP, GFLOPs, FPS, or deployment accuracy result is claimed here.

## Deployment

```bash
python -u tools/export_sarep_deploy.py --weights runs/sarep_spasff/sarep/weights/best.pt --output runs/sarep_spasff/sarep/weights/best_deploy.pt
python -u tools/benchmark_r18_sarep_spasff.py --weights runs/sarep_spasff/sarep/weights/best_deploy.pt --output runs/sarep_spasff/sarep/benchmark_results.json
```

The exporter compares training structure, in-memory fused structure, and independently reloaded deployment checkpoint. It refuses to publish the output if relative error exceeds 1e-4 **and** absolute error exceeds 1e-3. Benchmark only the verified deployment file when reporting deployed SA-Rep latency. Branch kernel norms are diagnostic statistics, not proof of detection benefit.

The complete-model unit test uses a randomly initialized RT-DETR. Its decoder selects queries with `torch.topk`; nearly equal random scores can change query order after tiny FP32 rounding differences from convolution fusion. The unit test therefore checks SA-Rep output, all three decoder input features, and encoder scores before top-k, then checks the full decoder output when query indices are unchanged. It also checks that the serialized deployment model reproduces the in-memory deployment output. The trained-checkpoint exporter retains a strict full-output check and will reject a checkpoint whose decoded predictions differ materially.

## Limits and checks still requiring server artifacts

- Verify original baseline `args.yaml`, logs, checkpoint, and dataset split against the new run before claiming a fair comparison.
- Train E1/E2 first; consider E3 only after their validation. Optional E1b/E1c/E5/E6 are prepared, not scheduled.
- Evaluate all models with the same checkpoint selection, split, input size and confidence handling. Re-run promising variants under at least three distinct seeds; a 0.001 single-seed AP change is not evidence of improvement.
- Check medium/large AP50:95 decline above 0.003, classification score shifts, box IoU, and latency. This is a screening threshold, not a statistical test.
