# 语音信号处理大作业 Track 1：Mini-BSRNN Baseline

这是“通用语音增强”课程赛题的可执行参考基线。任务说明见
[大作业 Track 1 PDF](docs/大作业%20track%201.pdf)。本仓库提供从数据清单、动态混合、四卡训练，
到 1000 条带真值验证、四项客观指标和复杂度统计的基本流程。

本项目是为课程作业独立整理的 Mini-BSRNN 教学基线，固定采用 16 kHz、64 维、2 层的
小规模结构。它用于跑通训练、推理与评测闭环，不兼容任何外部大型语音增强模型的权重。

## 1. 课程硬性规范

| 项目 | 本基线行为 |
|---|---|
| 输入 | 单通道、16 kHz、WAV格式 |
| 输出 | 单通道、16 kHz、WAV格式 |
| 长度 | 每个输出的采样点数与对应输入严格一致 |
| 验证 | 1000 对带真值音频，保存 clean/noisy/enhanced 对应关系 |
| 客观指标 | PESQ-WB、ESTOI、SI-SDR、UTMOS，逐文件计算后算术平均 |
| 无效结果 | 缺失、损坏、格式或长度错误时按最低值计分 |
| 效率验证 | batch size 1、单通道、16 kHz，报告参数量和 GMAC/s |

训练阶段使用 256 条内部样本计算 `val_loss` 并选择 checkpoint；最终客观指标统一在另行
提供的 1000 对 noisy/clean 音频上计算，保证不同模型使用相同输入和干净参考。

## 2. 模型

Mini-BSRNN 保留频带切分、时间/频率轴交替双向 LSTM，以及复数掩码加复数残差解码：
模型设计主要参考 BSRNN 在单通道语音增强中的两项工作 [1, 2]，本仓库在此基础上缩小
embedding 和循环层数，并固定为 16 kHz 教学配置。

| 配置 | 数值 |
|---|---:|
| 采样率 | 16 kHz |
| STFT / hop | 320 / 160 samples |
| 单边频点 | 161 |
| 频带数 | 28 |
| embedding | 64 |
| recurrent blocks | 2 |
| 参数量 | 2.154 M |
| 学习层 MAC/s | 2.537 GMAC/s |

`scripts/complexity.py` 的 MAC 定义覆盖所有 Conv1d、Linear 和双向 LSTM。STFT/ISTFT、
归一化和逐点操作因没有统一 MAC 映射而单独声明为未计入。若有队伍使用模型集成或外部预训练模块，必须把它们的参数和推理成本加入报告。

## 3. 环境

```bash
conda create -n course_track1 python=3.10 -y
conda activate course_track1
pip install -e '.[evaluation,test]'
```

UTMOS 第一次运行会从 `tarepan/SpeechMOS:v1.2.0` 下载课程指定的
`utmos22_strong` 权重，之后使用本地缓存。

## 4. 数据准备与声明

基线数据声明见
[docs/baseline_data_statement.json](docs/baseline_data_statement.json)：训练干净语音
17,396 条 / 28.8563 小时，内部验证语音 256 条 / 0.4736 小时；训练噪声
20,000 条 / 58.0314 小时，验证噪声 5,000 条 / 14.6528 小时；训练与验证 RIR
分别为 3,010 和 335 条。WHAM! 原始噪声是双声道，动态混合时明确取双声道均值转为
单声道；RIR 的 16/32/48 kHz 文件在使用时统一重采样到 16 kHz。

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

当前简化基线没有实现 Codec 与其他失真模拟；这是明确保留的改进方向，建议在训练中加入各种退化类型。

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

## 6. 已提供 checkpoint

仓库包含一次完整训练得到的参考权重：

```text
checkpoints/mini_bsrnn_best.ckpt
SHA256 632b3d0a8a3e9d27884a8fd2d500457211754fae3e0b37460463891c88c42aa9
```


该文件是去除优化器状态和训练配置对象后的可移植纯权重包（epoch 30、global step
16,500），可用于环境检查、推理接口自检和复现参考分数。

## 7. 1000 条带真值验证集上的完整验证

### 7.1 下载并校验

验证集包含 1000 对单声道 noisy/clean FLAC，总时长约 2.46 小时。原始音频包含
16、22.05、24、32、44.1 和 48 kHz 六种采样率，压缩包约 1.22 GB。

```bash
bash scripts/download_validation.sh
```

也可以从 [验证集下载页面](https://drive.google.com/file/d/1dPezrikPASvS2XfvceBF9VStflx3iVNj/view)
手动下载到 `data/downloads/validation_1000.zip`。脚本会核对压缩包 SHA256：

```text
edd77dccb6cc1d7c273f2a05a8daee0d26956bc748ac2472a1f9f7305a896080
```

### 7.2 准备固定 16 kHz 评测音频

Mini-BSRNN 固定使用 16 kHz，因此需要对 noisy 和对应 clean 使用同一种重采样规则，并
保持每一对音频严格等长。本仓库报告的是统一 16 kHz 后的课程指标，不能与直接在原始
多采样率音频上计算的数值混用：

```bash
python scripts/prepare_validation.py \
  --archive data/downloads/validation_1000.zip \
  --output-dir data/validation_1000
```

准备完成后会得到 `clean/`、`noisy/`、`clean.scp`、`noisy.scp` 和记录原始采样率分布及
转换方式的 `dataset_summary.json`。

### 7.3 增强并计算四项指标

```bash
DEVICE=cuda VALIDATION_BATCH_SIZE=1 bash scripts/validate.sh \
  checkpoints/mini_bsrnn_best.ckpt \
  data/validation_1000 \
  runs/validation_1000
```

命令会依次完成 1000 条 noisy 音频增强，以及 PESQ-WB、ESTOI、SI-SDR 和 UTMOS 计算。
输出包括：

- `runs/validation_1000/enhanced/`：模型输出 WAV；
- `runs/validation_1000/manifests/`：clean/noisy/enhanced 一一对应的 SCP；
- `runs/validation_1000/validation_run.json`：checkpoint 和验证数据记录；
- `runs/validation_1000/metrics/metrics.csv`：逐文件四项指标；
- `runs/validation_1000/metrics/summary.json` 和 `RESULTS.txt`：1000 条样本的指标均值。

调试流程时可以只跑第一条，并跳过需要额外模型的 UTMOS；该结果不能作为完整验证结果：

```bash
python scripts/prepare_validation.py --limit 1
DEVICE=cpu VALIDATION_LIMIT=1 bash scripts/validate.sh \
  checkpoints/mini_bsrnn_best.ckpt data/validation_1000 runs/validation_smoke \
  --skip-utmos
```

## 8. 参数量和计算量

```bash
python scripts/complexity.py \
  --duration 1.0 \
  --output logs/metrics_and_complexity/complexity.json
```

输出应连同命令和日志一起放入最终提交。该命令的输入口径为 batch size 1、
单通道、16 kHz、每秒 GMAC。

## 9. 建议实验产物

```text
team_<编号>/
├── README.md
├── src/
├── checkpoints/
├── runs/validation_1000/
├── logs/metrics_and_complexity/
└── team_<编号>_report.pdf
```

报告应覆盖摘要、全部数据来源与规模、划分和混合策略、模型与损失、验证指标、参数量、
GMAC/s、主要创新、局限和参考文献。

## 参考文献

1. J. Yu, H. Chen, Y. Luo, R. Gu, and C. Weng, “High Fidelity Speech
   Enhancement with Band-split RNN,” *Interspeech 2023*, pp. 2483–2487, 2023.
   [doi:10.21437/Interspeech.2023-1433](https://doi.org/10.21437/Interspeech.2023-1433)
2. J. Yu and Y. Luo, “Efficient Monaural Speech Enhancement with Universal
   Sample Rate Band-Split RNN,” *ICASSP 2023*, pp. 1–5, 2023.
   [doi:10.1109/ICASSP49357.2023.10096020](https://doi.org/10.1109/ICASSP49357.2023.10096020)

## License

代码按 [Apache License 2.0](LICENSE) 发布。数据集各自的许可不随本仓库授予。
