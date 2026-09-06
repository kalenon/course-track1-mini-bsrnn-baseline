# 语音信号处理大作业 Track 1：Mini-BSRNN Baseline

这是“通用语音增强”课程赛题的可执行参考基线。任务说明见
[大作业 Track 1 PDF](docs/大作业_track1.pdf)。本仓库提供从数据清单、动态混合、四卡训练，
到盲测批量推理、四项客观指标和复杂度统计的最小闭环。

本项目是为课程作业独立整理的 Mini-BSRNN 教学基线，固定采用 16 kHz、64 维、2 层的
小规模结构。它用于跑通训练、推理与评测闭环，不兼容任何外部大型语音增强模型的权重。

## 1. 课程硬性规范

| 项目 | 本基线行为 |
|---|---|
| 输入 | 单通道、16 kHz、WAV；不符合时直接报错 |
| 输出 | 单通道、16 kHz、PCM-16 WAV |
| 长度 | 每个输出的采样点数与对应输入严格一致 |
| 推理 | 支持目录递归和 SCP 清单，支持离线 batch |
| 客观指标 | PESQ-WB、ESTOI、SI-SDR、UTMOS，逐文件计算后算术平均 |
| 无效结果 | 缺失、损坏、格式或长度错误时按预先公布的最低值计分 |
| 效率口径 | batch size 1、单通道、16 kHz，报告参数量和 GMAC/s |

最终盲测只能使用冻结模型，提交 WAV 文件名和相对目录必须与输入一致。不要对盲测结果
做基于干净参考的后验对齐。

## 2. 模型

Mini-BSRNN 保留频带切分、时间/频率轴交替双向 LSTM，以及复数掩码加复数残差解码：

| 配置 | 数值 |
|---|---:|
| 采样率 | 16 kHz（固定） |
| STFT / hop | 320 / 160 samples |
| 单边频点 | 161 |
| 频带数 | 28 |
| embedding | 64 |
| recurrent blocks | 2 |
| 参数量 | 2,153,996（2.154 M） |
| 学习层 MAC/s | 2.537 GMAC/s |

`scripts/complexity.py` 的 MAC 定义覆盖所有 Conv1d、Linear 和双向 LSTM。STFT/ISTFT、
归一化和逐点操作因没有统一 MAC 映射而单独声明为未计入；所有队伍比较时必须使用同一
脚本与口径。若使用模型集成或外部预训练模块，必须把它们的参数和推理成本加入报告。

## 3. 环境

推荐 Python 3.10：

```bash
conda create -n course_track1 python=3.10 -y
conda activate course_track1
pip install -e '.[evaluation,test]'
```

UTMOS 第一次运行会从 `tarepan/SpeechMOS:v1.2.0` 下载课程指定的
`utmos22_strong` 权重，之后使用本地缓存。

## 4. 数据准备与声明

本次参考训练实际使用：

- 干净语音：TIMIT `train` 与 WSJ0 `si_tr_s`；
- 内部验证语音：WSJ0 `si_dt_05` 中等距选取 256 条；
- 噪声：WHAM! `tr` / `cv` 分别用于训练和验证；
- RIR：`record_RIR_with_T60_distance` 按稳定文件序列做 90/10 划分；
- 训练时统一重采样到 16 kHz，片段长度 32,000 samples（2 秒）；
- SNR 为 -5 至 20 dB，混响概率 0.5，并随机加入带宽限制、削波和丢包。

本机实际扫描得到的基线数据声明见
[docs/baseline_data_statement.json](docs/baseline_data_statement.json)：训练干净语音
17,396 条 / 28.8563 小时，内部验证语音 256 条 / 0.4736 小时；训练噪声
20,000 条 / 58.0314 小时，验证噪声 5,000 条 / 14.6528 小时；训练与验证 RIR
分别为 3,010 和 335 条。WHAM! 原始噪声是双声道，动态混合时明确取双声道均值转为
单声道；RIR 的 16/32/48 kHz 文件在使用时统一重采样到 16 kHz。

TIMIT 和 WSJ 需要合法授权，本仓库不重新分发任何语料。没有授权时请改用题目 PDF 中的
公开候选语料，并在报告中如实披露名称、版本、规模、清洗与划分规则。

在本工作区复现清单的命令如下；其他机器只需替换四个根目录：

```bash
python scripts/prepare_data.py \
  --timit-root /data/Database/clean/speech/TIMIT \
  --wsj-root /data/Database/clean/speech/WSJ \
  --wham-root /data/Database/noise/wham \
  --rir-root /data/Database/RIR/record_RIR_with_T60_distance \
  --output-dir data/manifests \
  --validation-size 256
```

命令同时生成 `data/manifests/data_statement.json`，其中包含每个划分的条目数、时长、
采样率和声道数，应作为技术报告数据声明的依据。SCP 格式为：

```text
utterance_id /absolute/path/to/audio.wav
```

当前简化基线没有实现 Codec 与专用风噪模拟；这是明确保留给学生改进的方向，而不是在
报告中可以省略的退化类型。

## 5. 四卡训练

```bash
bash scripts/train_4gpu.sh
```

也可以显式指定 GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python scripts/train.py --config configs/mini_bsrnn.yaml --devices 4
```

训练使用动态混合、AdamW、稳定的多分辨率 L1 频谱损失，按 `val_loss` 保存最优模型。
日志和权重位于 `runs/mini_bsrnn/`。监控曲线：

```bash
tensorboard --logdir runs/mini_bsrnn
```

断点继续：

```bash
python scripts/train.py --config configs/mini_bsrnn.yaml \
  --devices 4 --resume runs/mini_bsrnn/checkpoints/last.ckpt
```

## 6. 已提供 checkpoint

仓库包含一次完整训练得到的参考权重：

```text
checkpoints/mini_bsrnn_best.ckpt
SHA256 632b3d0a8a3e9d27884a8fd2d500457211754fae3e0b37460463891c88c42aa9
```

验证哈希：

```bash
sha256sum -c CHECKSUMS.sha256
```

该文件是去除优化器状态和训练配置对象后的可移植纯权重包（epoch 30、global step
16,500），可用于环境检查、推理接口自检和复现参考分数。

## 7. 严格 16 kHz 批量推理

盲测目录推理（递归保持文件名和相对目录）：

```bash
bash scripts/infer.sh \
  /path/to/blind_test \
  enhanced_test \
  checkpoints/mini_bsrnn_best.ckpt \
  --device cuda --batch-size 1
```

也可以通过 `--input-scp path/to/wav.scp` 输入多个文件。输出清单固定写到
`enhanced_test/enhanced.scp`。脚本不会把 16 kHz 输出再次上采样，也默认不做峰值归一化。

推理后可快速检查：

```bash
python - <<'PY'
from pathlib import Path
import soundfile as sf
for path in Path('enhanced_test').rglob('*.wav'):
    info = sf.info(path)
    assert info.samplerate == 16000 and info.channels == 1
print('format check passed')
PY
```

## 8. 验证集四项客观指标

准备干净参考清单 `reference.scp` 和推理生成的 `enhanced.scp`，两者 ID 必须一致：

```bash
bash scripts/evaluate.sh \
  data/validation/reference.scp \
  enhanced_validation/enhanced.scp \
  logs/metrics_and_complexity/validation \
  --device cuda
```

输出包括：

- `metrics.csv`：逐文件 PESQ、ESTOI、SI-SDR、UTMOS 和有效性状态；
- `summary.json`：样本数、惩罚样本数、均值及惩罚值；
- `RESULTS.txt`：便于粘贴进报告的总体结果。

默认无效文件惩罚为 PESQ=-0.5、ESTOI=0、SI-SDR=-50 dB、UTMOS=1。课程组若发布新的
统一下限，应以课程组口径为准，并对所有系统一致重算。

## 9. 参数量和计算量

```bash
python scripts/complexity.py \
  --duration 1.0 \
  --output logs/metrics_and_complexity/complexity.json
```

输出应连同命令和日志一起放入最终提交。该命令的输入口径正是 PDF 要求的 batch size 1、
单通道、16 kHz、每秒 GMAC。

## 10. 自检

```bash
python -m unittest discover -s tests -v
python scripts/train.py --config configs/mini_bsrnn.yaml --devices 1 --fast-dev-run
```

第二条命令需要先完成数据清单准备。第一条不需要语料或 GPU。

## 11. 推荐提交结构

```text
team_<编号>/
├── README.md
├── src/
├── scripts/infer.sh
├── checkpoints/
├── enhanced_test/
├── logs/metrics_and_complexity/
└── team_<编号>_report.pdf
```

报告应覆盖摘要、全部数据来源与规模、划分和混合策略、模型与损失、验证指标、盲测结果、
参数量、GMAC/s、主要创新、局限和参考文献。主观测听由课程组匿名化系统并随机化样例后
组织，Track 1 样本由 Track 2 同学交叉评价。

## License

代码按 [Apache License 2.0](LICENSE) 发布。数据集各自的许可不随本仓库授予。
