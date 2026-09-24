# Track 1 基线复现与新版指标

赛题说明以 `docs/大作业 track 1.md` 为准。本仓库已有数据清单准备、Mini-BSRNN 训练、参考权重和 1000 对带标签验证音频的下载脚本。

## 数据和训练

```bash
conda create -n ssp-track1 python=3.10 -y
conda activate ssp-track1
pip install -e '.[evaluation]'
pip install -r requirements-course-evaluation.txt
python scripts/prepare_data.py \
  --libritts-root /path/to/LibriTTS \
  --wham-root /path/to/WHAM48kHz \
  --slr26-root /path/to/DNS_ICASSP2021/datasets/impulse_responses/SLR26 \
  --slr28-root /path/to/DNS_ICASSP2021/datasets/impulse_responses/SLR28 \
  --output-dir data/manifests --validation-size 256
python scripts/train.py --config configs/mini_bsrnn.yaml --devices 1
```

清单构建使用 LibriTTS `train-*` / `dev-*`、WHAM! `tr` / `cv`，
以及 DNS RIR SLR26+SLR28 的固定文件划分；源音频会在训练读取时
重采样到 16 kHz。

候选数据下载位置与许可要求见课程赛题 Markdown。训练命令生成 `runs/mini_bsrnn/` 下的 checkpoint。本仓库提供的 `checkpoints/mini_bsrnn_best.ckpt` 可直接用于推理自检。

## 固定验证集与客观分数

```bash
bash scripts/download_validation.sh
python scripts/prepare_validation.py --archive data/downloads/validation_1000.zip \
  --output-dir data/validation_1000
python scripts/infer.py --input-dir data/validation_1000/noisy \
  --output-dir runs/validation_1000/enhanced \
  --checkpoint checkpoints/mini_bsrnn_best.ckpt --device cuda
bash scripts/download_metric_weights.sh checkpoints/metric_weights
python scripts/evaluate_course.py \
  --reference-dir data/validation_1000/clean \
  --enhanced-dir runs/validation_1000/enhanced \
  --dnsmos-primary checkpoints/metric_weights/sig_bak_ovr.onnx \
  --dnsmos-p808 checkpoints/metric_weights/model_v8.onnx \
  --output-dir runs/validation_1000/course_metrics --device cuda
python scripts/complexity.py --duration 1.0 --output logs/metrics_and_complexity/complexity.json
```

评分脚本输出逐文件 CSV 和四项均值 JSON。DNSMOS 与 UTMOS 首次安装或运行时需要下载权重。`--skip-nonintrusive` 只能做格式与有参指标自检，不是正式成绩。

## 盲测接口

```bash
python scripts/infer.py --input-dir /path/to/blind_noisy \
  --output-dir runs/blind/enhanced \
  --checkpoint checkpoints/mini_bsrnn_best.ckpt --device cuda
python scripts/validate_submission.py --input-dir /path/to/blind_noisy \
  --output-dir runs/blind/enhanced
```

盲测只需损伤语音，不需要干净参考。输出目录镜像输入目录的 WAV 文件名，16 kHz 单通道且采样点数严格相同。

## 基线成绩

参考权重在 1000 对固定验证集上的实测值见 [BASELINE_RESULTS.md](BASELINE_RESULTS.md)：PESQ-WB 1.4197、ESTOI 0.6732、DNSMOS-OVRL 2.1696、UTMOS 1.6753；1000 条均有效。精确数值与逐文件结果保存在 `runs/validation_1000/course_metrics/`。
