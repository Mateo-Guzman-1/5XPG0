# Encoding: comparison of clip and continuous-listening evaluation

All 61 encodings were evaluated with three matched seeds using the original compact SNN training recipe. The same integer checkpoints were then scored at the clip-F1 threshold and at Pedro R1’s validation-selected streaming threshold. Numbers below are mean ± sample SD across seeds. The detailed per-model results and margins are saved with the run.

Negative audio: 18.109 validation hours and 18.965 test hours. Live tests use 212 Sheila and 3,000 other-word clips, randomly placed in background noise. The standalone validation limits are 0.2% other-word acceptance and 2 false accepts/hour.

| Baseline metric | Result |
|---|---:|
| Isolated-clip accuracy (%) | 98.45 ± 0.10 |
| Isolated-clip F1 (%) | 87.86 ± 0.72 |
| Full 11,005-clip test F1 (%) | 74.78 ± 2.33 |
| Live recall, clip-F1 threshold (%) | 70.91 ± 3.57 |
| FA/h, clip-F1 threshold | 149.87 |
| Live recall, streaming threshold (%) | 32.86 ± 2.37 |
| FA/h, streaming threshold | 3.88 |
| Input / parameters / FPGA inference | 1024 bits / 6440 bytes / 10.40 ms |

## Sweep comparison

Each row shows the setting selected on mean validation recall within that sweep. Test values describe that fixed choice; they were not used to select it.

| Variable | Validation-selected setting | Clip accuracy (%) | Clip F1 (%) | Live recall (%) | FA/h | Input bits | Parameters (B) | FPGA (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| time_bins | time_bins=32 | 98.56 ± 0.08 | 88.83 ± 0.62 | 35.06 ± 2.23 | 5.69 | 2048 | 12584 | 17.56 |
| activity | rms_gate_u6=5000, rms_full_scale_u6=20000 | 98.59 ± 0.21 | 89.02 ± 1.30 | 33.65 ± 3.41 | 3.90 | 1024 | 6440 | 10.40 |
| band_mapping | band_mapping=equal | 98.45 ± 0.10 | 87.86 ± 0.72 | 32.86 ± 2.37 | 3.88 | 1024 | 6440 | 10.40 |
| fft | fft_window_ms=15, fft_hop_ms=10 | 98.56 ± 0.21 | 88.88 ± 1.56 | 31.60 ± 2.06 | 4.08 | 1024 | 6440 | 10.41 |
| quantizer | bits=4, quantizer=log | 98.30 ± 0.08 | 86.71 ± 0.42 | 32.39 ± 1.91 | 4.13 | 512 | 6440 | 10.40 |
| feature_mode | feature_mode=absolute_delta | 98.51 ± 0.08 | 88.08 ± 0.55 | 32.86 ± 5.99 | 3.46 | 2048 | 12584 | 17.57 |
| bits | bits=6 | 98.37 ± 0.04 | 87.16 ± 0.37 | 31.92 ± 2.72 | 4.45 | 768 | 6440 | 10.40 |
| bands | bands=20 | 98.52 ± 0.06 | 88.39 ± 0.82 | 35.85 ± 2.45 | 4.73 | 2560 | 15656 | 21.18 |
| range | low_hz=300, high_hz=3400 | 98.45 ± 0.10 | 87.86 ± 0.72 | 32.86 ± 2.37 | 3.88 | 1024 | 6440 | 10.40 |
| duration | duration_ms=800 | 98.38 ± 0.24 | 87.64 ± 1.28 | 34.43 ± 1.25 | 4.87 | 1024 | 6440 | 10.38 |

## Interpretation

Clip F1 and live recall have different denominators and operating points, so their difference is not an accuracy loss in percentage points. The paired old/new threshold comparison above isolates the threshold trade-off on the same streaming audio. Changing the evaluation protocol does not change model memory or input bits. FPGA latency is measured separately; acoustic latency is conditional on successfully detected words.

The duration sweep retains centre cropping/zero padding inside a one-second request. Longer durations add no new acoustic context. Seed SD does not quantify uncertainty across speakers or rooms. The offline streaming measurements do not constitute a physical microphone-to-LED test.

Validation-selected configuration: `b8_f8_tb16_hz300-3400_t800_g3000_s20000_mapequal_w25_h10_qlinear_mabsolute`.
Board checks: 2440 vectors; 0 mismatches.
