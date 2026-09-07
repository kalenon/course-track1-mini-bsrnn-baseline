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

## Labeled validation preparation and inference

The 1000-pair archive checksum, ZIP CRC, utterance IDs and original sample rates
are checked before preparation. A 48-kHz noisy/clean pair was resampled to 16 kHz
and enhanced on CPU with the included tensor-only checkpoint. The clean, noisy
and enhanced outputs were verified as mono, 16-kHz WAV files with matching frame
counts.

## Metrics smoke test

PESQ-WB, ESTOI and SI-SDR were run on the enhanced smoke-test file and
`num_penalized` was zero. UTMOS had already been tested separately with the same
evaluation entry point. A one-file smoke test is an execution check, not a
meaningful quality benchmark; the reported course result must use all 1000 pairs.

## Complexity

```bash
python scripts/complexity.py --duration 1.0
```

Result: 2,153,996 parameters and 2.537417344 learned-layer GMAC/s for batch size
1, one 16-kHz mono second. See the definition emitted in the JSON report.
