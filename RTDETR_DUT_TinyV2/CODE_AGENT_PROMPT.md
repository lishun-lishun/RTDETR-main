# Code Agent任务：对DUT-Anti-UAV的四项RT-DETR训练与Query创新进行真实工程集成和对照实验

项目：`https://github.com/lishun-lishun/RTDETR-main`。这是Ultralytics YOLOv8.0.201分支，不是D-FINE官方代码；训练环境Python3.10、torch2.3.1+cu121、A30 24GB，输入640，DUT-Anti-UAV测试2200张/2245目标。用户原始R18: AP50=.966, AP75=.816, AP50:95=.707；HRNet历史AP50:95=.723。用户已试BAPS、RDCF、CSP、LDW、R01-R16和上一版P2局部证据采样，均未产生稳定提升。禁止重新引入之前无效的P2采样或通用Backbone堆叠。

## 1. 代码基准与已提供模块

读取当前实际Git Commit，并与包说明核对。本包包含：`src/tiny_uav_decoder.py`（M1 LPQ、M2 PDN）、`src/tiny_uav_loss.py`（M3 TCM、M4 SNWD）、`install_tiny_v2.py`、`run_tiny_v2.py`、`tests/`。请阅读完整源码及安装器，先`--repo ...` dry-run，核查两处补丁，再`--apply`。原`rtdetr-r18.yaml`、历史权重、runs必须保持不变。不要复制完整工程。模型更新仅通过新配置：`rtdetr_r18_tinyv2_*.yaml`。

真实代码接口：`RTDETRDecoder`定义在`ultralytics/nn/modules/head.py`，输出`dec_bboxes,dec_scores,enc_bboxes,enc_scores,dn_meta`；其`_get_decoder_input`负责Encoder Top-K Query；训练DN使用`ultralytics/models/utils/ops.py::get_cdn_group`。`RTDETRDetectionModel`定义于`ultralytics/nn/tasks.py`，Loss来自`ultralytics/models/utils/loss.py`。`parse_model()`需要识别`TinyUAVDecoder`并注入通道列表。原始R18最后一层`[[19,22,25],1,RTDETRDecoder,[nc,256,300,4,8,3]]`。

## 2. 四个研究机制

- **M1 LPQ**：参考DINO混合Query初始化，用每个P3/P4/P5尺度内的局部峰值响应小幅修正Encoder分类分数的Query排序，仍选300 Query，不新增第四特征层，不采用固定SAQS额度。机制：`score'_i=score_i+alpha*clamp(sigmoid(score_i)-local_avg(...),0,.5)`。对比alpha=0、0.3、0.6，统计GT Query覆盖和各尺度来源。
- **M2 PDN**：参考DINO对比去噪，保持原DN组、mask、正/负样本索引，在训练期按像素级尺度生成可控位置噪声，避免tiny框的相对噪声过大；必须核查负样本真的足够难、正样本不会越界；对照原始DN与各像素抖动幅度。
- **M3 TCM**：在原Hungarian class+L1+GIoU Cost上增加微小GT中心相对距离：`lambda * sigmoid((T-max(w,h)_px)/tau) * (center_distance_px / diag_gt_px)/(1+center_distance_px/diag_gt_px)`。对照原Matcher、uniform center cost和tiny-conditioned center cost。主分支仍One-to-One，无测试GT。
- **M4 SNWD**：参考NWD/2026年DUT-ANTI-UAV的UAV-DETR，保留原L1+GIoU，在小GT上添加尺度调制NWD：`lambda * mean(sigmoid((T-size_px)/tau)*(1-exp(-W2px/C)))`。检查多Decoder辅助、Encoder层和DN损失；与仓库内已有普通NWD机制进行直接消融，避免将现有NWD包装成新发明。

严禁把四点称为已证实的新颖贡献。与2026年同数据集UAV-DETR (arxiv:2603.22841, github.com/wd-sir/UAVDETR)、DEIM CVPR 2025、RT-DETRv3 WACV 2025、DINO ICLR 2023、NWD AI-TOD论文逐项比较，并将重复/已有工作如实指出。至少报告两项能够解释的核心科研假设，不要强求四点都写入论文。

## 3. 实际工程校验

1. 测试模块class继承、parameter/state_dict键兼容。对`off_control`和原始R18载入同权重，比对eval输出在容差内一致；train路径固定随机seed，并排查浮点与随机数消费差异。
2. 测试CPU与A30 CUDA，Batch1/2，含0/1/多GT的训练，AMP/FP32，Grad、NaN/Inf、梯度流与优化器参数更新；保存/重新加载权重后重新评估。
3. 验证M2只修改DN参考框，DN索引/负样本组隔离与注意力mask不变；当M2关闭时调用原始`get_cdn_group`逻辑。
4. 验证M3 GT分组偏移、CPU scipy匹配、无GT、双图与不同GT数量情况下索引正确；禁止预测框泄露GT至推理。
5. 验证M4的像素归一化真实正确：1px位移、完全重合、极小box、非重叠、梯度有界；确认loss不重复叠加仓库原有NWD分支。
6. `TinyUAVDecoder`有一个细节：只有启用M2并处于train时才走覆写forward，否则由原始head.forward执行；但M1通过`_get_decoder_input`的动态分派仍然生效。检查这一点。
7. 执行`python -m pytest -q <代码包路径>/tests/test_tinyv2.py`；修复测试中真正暴露的问题，不允许绕过测试。

## 4. 实验协议

先审计历史warmup_epochs=2000、EMA、优化器、增强、Train/Val/Test划分与跨split重复图片问题。统一Baseline R18后，第一批单因素：M1、M2、M3、M4；第二批：M1+M3、M2+M4；只有单项出现稳定迹象才运行all4。不在测试集选超参数。短程40ep必须有同协议40ep原始基线；优选完整150ep与3个随机种子。要求保存正式COCO AP50/AP75/AP50:95、Small/Medium/Large、精确分组Tiny(最大边≤8)、Small(8-16)、Medium(16-32)、Large(>32)的自定义AP及GT数量。完整Batch1/640/FP16/A30 CUDA同步延迟与FPS，禁止混用GFLOPs工具。

## 5. 交付与决策

给出：实际代码改动列表、可复现的安装/训练/评估命令、8份实验配置、模型/匹配/损失的单元测试、失败实验原因、四种方法单独和联合消融表、与真实2026论文新颖性矩阵。实现有效不代表论文创新；对重复、无收益模块明确淘汰。禁止覆盖原始文件，禁止直接启动完整训练，所有未运行指标标注“未运行”。
