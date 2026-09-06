# Baseline verification

The following checks were run before publishing this teaching repository.

## Model tests

```bash
python -m unittest discover -s tests -v
```

Result: 3/3 tests passed. The checks cover the fixed 16-kHz, 64-dimensional,
two-block architecture, exact 2,153,996 parameter count, waveform length and
padding behavior, finite model output, and finite loss/gradients near silence.

## Training smoke test

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --config configs/mini_bsrnn.yaml --devices 1 --fast-dev-run
```

Result: one real dynamic-mixing training batch and one deterministic validation
batch completed on GPU without non-finite values.

## Portable checkpoint and inference

The included checkpoint contains tensor state only; it does not pickle training
configuration classes. A 16-kHz mono PCM-16 WAV was enhanced on CPU with
the standalone repository. The output was verified as mono, 16 kHz, PCM-16 WAV,
with exactly the same number of frames as the input.

## Metrics smoke test

PESQ-WB, ESTOI, SI-SDR and UTMOS were run on the enhanced smoke-test file. All
four metrics completed and `num_penalized` was zero. This is an execution check,
not a meaningful quality benchmark.

## Complexity

```bash
python scripts/complexity.py --duration 1.0
```

Result: 2,153,996 parameters and 2.537417344 learned-layer GMAC/s for batch size
1, one 16-kHz mono second. See the definition emitted in the JSON report.
