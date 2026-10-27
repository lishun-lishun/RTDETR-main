"""Generate CSP variants by replacing only Stage3/4 lines of original R18."""
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parents[1] / 'ultralytics/cfg/models/rt-detr'
BASE = (MODEL_DIR / 'rtdetr-r18.yaml').read_text(encoding='utf-8')
ORIGINAL_STAGE3 = "  - [-1, 1, Blocks, [128, BasicBlock, 2, 3, 'relu']] # 5-P3/8"
ORIGINAL_STAGE4 = "  - [-1, 1, Blocks, [256, BasicBlock, 2, 4, 'relu']] # 6-P4/16"
CONFIGS = {
    'stage3': ('rtdetr_r18_csp_stage3.yaml', 0.75, None),
    'stage4': ('rtdetr_r18_csp_stage4.yaml', None, 7),
    'combined': ('rtdetr_r18_csp_context.yaml', 0.75, 7),
    'stage4_k3': ('rtdetr_r18_csp_stage4_k3.yaml', None, 3),
    'stage3_ratio05': ('rtdetr_r18_csp_stage3_ratio05.yaml', 0.5, None),
}


def generate(name):
    _, ratio, kernel = CONFIGS[name]
    result = BASE
    if ratio is not None:
        result = result.replace(ORIGINAL_STAGE3, (
            f'  - [-1, 1, CSPResidualStage, [128, 2, 3, {ratio}]] # 5-P3/8'))
    if kernel is not None:
        result = result.replace(ORIGINAL_STAGE4, (
            f'  - [-1, 1, CSPContextStage, [256, 2, 4, 0.75, {kernel}]] # 6-P4/16'))
    if result == BASE or (ratio is not None and ORIGINAL_STAGE3 in result) or (
            kernel is not None and ORIGINAL_STAGE4 in result):
        raise RuntimeError('original R18 stage line changed')
    if result.split('head:\n', 1)[1] != BASE.split('head:\n', 1)[1]:
        raise RuntimeError('head changed during CSP YAML generation')
    return result


if __name__ == '__main__':
    for name, (filename, _, _) in CONFIGS.items():
        path = MODEL_DIR / filename
        path.write_text(generate(name), encoding='utf-8')
        print(path)
