"""Derive Res2-BasicBlock and Lite-GDNeck variants from untouched R18 YAML."""
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parents[1] / 'ultralytics/cfg/models/rt-detr'
BASE = (MODEL_DIR / 'rtdetr-r18.yaml').read_text(encoding='utf-8')
VARIANTS = {
    'res2basic': ([(3, [1])], 4, True, None),
    'res2basic_scale2': ([(3, [1])], 2, True, None),
    'res2basic_both': ([(3, [0, 1])], 4, True, None),
    'res2basic_stage34': ([(3, [1]), (4, [1])], 4, True, None),
    'res2basic_independent': ([(3, [1])], 4, False, None),
    'litegd': ([], 4, True, 'full'),
    'litegd_high_only': ([], 4, True, 'high_only'),
    'litegd_low_only': ([], 4, True, 'low_only'),
    'litegd_local_only': ([], 4, True, 'local_only'),
    'litegd_p3_only': ([], 4, True, 'p3_only'),
    'res2basic_litegd': ([(3, [1])], 4, True, 'full'),
}


def generate(name):
    stages, scale, hierarchical, neck_mode = VARIANTS[name]
    result = BASE
    for stage_num, indices in stages:
        channels = {3: 128, 4: 256}[stage_num]
        old = f"- [-1, 1, Blocks, [{channels}, BasicBlock, 2, {stage_num}, 'relu']]"
        new = (f"- [-1, 1, Res2Blocks, [{channels}, 2, {stage_num}, 'relu', "
               f"{indices}, {scale}, null, {str(hierarchical).lower()}]]")
        if old not in result:
            raise RuntimeError(f'baseline stage {stage_num} changed')
        result = result.replace(old, new)
    if neck_mode:
        prefix, original_head = result.split('head:\n', 1)
        first_two = original_head.splitlines()[:2]
        if len(first_two) != 2 or 'AIFI' not in first_two[1]:
            raise RuntimeError('baseline P5 projection/AIFI changed')
        result = prefix + 'head:\n' + '\n'.join(first_two) + '\n\n' + '\n'.join([
            f"  - [[5, 6, 9], 1, LiteGDNeck, [128, 256, '{neck_mode}']]  # 10: F3/F4/F5",
            '  - [10, 1, GDFeature, [0]]  # 11 F3',
            '  - [10, 1, GDFeature, [1]]  # 12 F4',
            '  - [10, 1, GDFeature, [2]]  # 13 F5',
            '  - [[11, 12, 13], 1, RTDETRDecoder, [nc, 256, 300, 4, 8, 3]]',
        ]) + '\n'
    return result


if __name__ == '__main__':
    for name in VARIANTS:
        path = MODEL_DIR / f'rtdetr_r18_{name}.yaml'
        path.write_text(generate(name), encoding='utf-8')
        print(path)
