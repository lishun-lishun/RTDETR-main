"""Generate checked-in SA-Rep/SP-ASFF ablations from the untouched R18 YAML."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / 'ultralytics/cfg/models/rt-detr'
BASE = (ROOT / 'rtdetr-r18.yaml').read_text(encoding='utf-8')
VARIANTS = {
    'sarep': ((5,), (1,), None),
    'sarep_both': ((5,), (0, 1), None),
    'sarep_stage34': ((5, 6), (1,), None),
    'spasff': ((), (), 'protected'),
    'sarep_spasff': ((5,), (1,), 'protected'),
    'asff_direct': ((), (), 'direct'),
    'spasff_no_difference': ((), (), 'no_difference'),
}


def generate(name):
    stages, indices, mode = VARIANTS[name]
    result = BASE
    for stage in stages:
        channels, stage_num = (128, 3) if stage == 5 else (256, 4)
        original = f"- [-1, 1, Blocks, [{channels}, BasicBlock, 2, {stage_num}, 'relu']]"
        replacement = (f"- [-1, 1, SARepBlocks, "
                       f"[{channels}, 2, {stage_num}, 'relu', {list(indices)}]]")
        if original not in result:
            raise ValueError(f'baseline stage {stage} changed')
        result = result.replace(original, replacement)
    if mode:
        decoder = '- [[19, 22, 25], 1, RTDETRDecoder, [nc, 256, 300, 4, 8, 3]]'
        if decoder not in result:
            raise ValueError('baseline decoder indices changed')
        result = result.replace(
            decoder,
            f"- [[19, 22, 25], 1, SPASFF, [64, '{mode}']]  # 26 final F3 only\n"
            "  - [[26, 22, 25], 1, RTDETRDecoder, [nc, 256, 300, 4, 8, 3]]")
    return result


if __name__ == '__main__':
    for variant in VARIANTS:
        path = ROOT / f'rtdetr_r18_{variant}.yaml'
        path.write_text(generate(variant), encoding='utf-8')
        print(path)
