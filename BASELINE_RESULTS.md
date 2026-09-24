# Track 1 参考基线验证成绩

以下是本仓库 Mini-BSRNN 参考权重在固定 1,000 对验证音频上，按课程当前四项指标测得的结果。所有音频先统一为单通道 16 kHz，并保持 noisy、clean、enhanced 严格等长。结果是验证集成绩，不是正式排名的盲测成绩。

| 指标 | 实测均值 |
|---|---:|
| PESQ-WB | 1.4197 |
| ESTOI | 0.6732 |
| DNSMOS-OVRL | 2.1696 |
| UTMOS | 1.6753 |

评测文件数：1,000；缺失、格式错误等被惩罚文件数：0。参考模型参数量 2,153,996；学习层计算量 2.5374 GMAC/s（batch size 1、16 kHz；STFT/ISTFT 等非学习算子未计入），由 `scripts/complexity.py` 实测。逐文件结果和完整精度汇总在本地运行产生的 `runs/validation_1000/course_metrics/per_file.csv`、`summary.json` 中。

复现标识：

- Mini-BSRNN 参考权重：`checkpoints/mini_bsrnn_best.ckpt`，随本仓库代码提交；
- 固定验证集原始压缩包：`data/downloads/validation_1000.zip`，通过 `scripts/download_validation.sh` 下载；
- PESQ-WB、ESTOI 由 `pesq`、`pystoi` 计算；DNSMOS-OVRL 使用 `scripts/download_metric_weights.sh` 下载的权重；UTMOS 使用 `tarepan/SpeechMOS:v1.2.0` 的 `utmos22_strong` 权重;
- 完整复现命令见 [COURSE_GUIDE.md](COURSE_GUIDE.md)。

上述四项是原始指标值。课程总评分还包括效率、主观评价和报告评分，评分细则见 `docs/大作业 track 1.md`。
