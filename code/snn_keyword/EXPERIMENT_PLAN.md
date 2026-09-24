# Group 2 experiment plan (before training)

Question: can a 768-input, 64-LIF-neuron network detect the word **yes**
with useful precision/recall while fitting the 256 KiB PicoRV32 BRAM and
finishing inference within a 250 ms audio hop at an assumed 100 MHz?

Hypothesis: caching the constant input projection will reduce processor
cycles substantially compared with recomputing a projection for each rate
encoding step, without a large loss of detection F1.

Hold constant: Speech Commands v0.02 official speaker-disjoint split;
24x32 mel frontend; 64 hidden neurons; 12 time steps; beta=7/8;
immediate subtractive reset; two-class nonspiking accumulated readout;
AdamW; 35 epochs; 16,384 balanced sampled examples per epoch; three seeds.
The last ten epochs use fake quantization to the deployment Q10 grid.
Select checkpoints and F1 thresholds on validation data only. Select the
deployment model by validation F1, breaking ties in favor of current encoding.
Evaluate each final checkpoint once on the complete official test set.

Report confusion matrices, precision, recall, F1, negative-clip FPR,
across-seed variability, quantization disagreement, memory footprint,
actual RTL-simulated processor cycles, and hidden spike counts. MACs and
spike operations are cost proxies, not measured energy. Clip FPR is not
false accepts per hour on continuous speech.

Verification: independent NumPy integer oracle vs native C on the entire
test set; the same C compiled for RV32IM vs PicoRV32 RTL on positive,
negative, silence, saturation, random, repeated, and malformed requests;
mailbox sequence handling; LED timing and 32-bit timer wrap; TCP replay.
Deployment selection applies the 250 ms constraint before ranking validation
F1: the unconstrained validation winner is retained as an experiment, but
an encoding that exceeds the measured cycle budget cannot be the live demo.
This engineering selection uses timing and validation data, not test accuracy.

No board measurement or physical timing/energy claim is possible without
the PYNQ. Keep that distinction in the report and presentation.

Sources: [Speech Commands paper](https://arxiv.org/abs/1804.03209),
[dataset](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz),
[PicoRV32](https://github.com/YosysHQ/picorv32).
