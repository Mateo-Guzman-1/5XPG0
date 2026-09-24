"""Render the scientific report and presentation from measured JSON results."""
import json
from pathlib import Path
import textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT=Path(__file__).resolve().parent


def main():
    read=lambda n:json.loads((ROOT/'results'/n).read_text())
    e,v,h,b=map(read,['evaluation.json','verification.json','hardware.json','background_noise.json'])
    chosen=next(r for r in e['runs'] if r['selected']); m=chosen['test']
    rate=read('verification_rate.json')['rtl']; cur=v['rtl']
    by={k:[r['test']['f1'] for r in e['runs'] if r['encoding']==k] for k in ['current','rate']}
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    axes[0].bar(['Current','Rate'],[100*np.mean(by[k]) for k in by],
                yerr=[100*np.std(by[k],ddof=1) for k in by],capsize=6,color=['#197d96','#e49b35'])
    axes[0].set(ylabel='Test F1 (%)',ylim=(0,100),title='Three seeds: mean and sample SD')
    axes[1].bar(['Current','Rate'],[cur['milliseconds_max_at_100mhz'],rate['milliseconds_max_at_100mhz']],color=['#197d96','#e49b35'])
    axes[1].axhline(250,color='black',ls='--',label='250 ms hop')
    axes[1].set(ylabel='Worst observed inference (ms)',title='40 RTL vectors at 100 MHz');axes[1].legend()
    fig.savefig(ROOT/'results/comparison.png',dpi=180)
    table='\n'.join(f'| {r["encoding"]} | {r["seed"]} | {r["validation"]["f1"]:.4f} | {r["test"]["precision"]:.4f} | {r["test"]["recall"]:.4f} | {r["test"]["f1"]:.4f} | {100*r["test"]["fpr"]:.3f}% |' for r in e['runs'])
    report=f'''# Group 2: spiking keyword detection on PicoRV32

## Research question and outcome

Can an integer SNN detect **yes** while fitting the PYNQ-Z2's 256 KiB
RISC-V BRAM and completing inference inside a 250 ms streaming hop?
The hypothesis was that caching a constant spectrogram projection would
reduce latency substantially relative to rate encoding without a large
detection-quality penalty. The results support that hypothesis for this
fixed 768–64–2 topology and this dataset; they do not establish a general
advantage of one encoding or SNNs over ANNs.

The deployed model is **{e['selected_run']}**: precision {100*m['precision']:.2f}%,
recall {100*m['recall']:.2f}%, F1 {100*m['f1']:.2f}%. Its maximum observed RTL
latency was **{cur['milliseconds_max_at_100mhz']:.2f} ms** at 100 MHz.
No physical board or microphone measurement was performed.

## Dataset, controls, and training

Speech Commands v0.02 contains 105,829 isolated spoken-word clips. The
official split supplies 84,843 training, 9,981 validation, and 11,005 test
clips; positive counts are 3,228, 397, and 419. Speaker sets are checked for
disjointness. All 34 non-keyword words form the negative class. The archive
SHA256 and source URL are recorded in `results/dataset.json`.

Each encoding was trained with seeds 0, 1, and 2, AdamW (initial learning
rate 0.002, cosine decay, weight decay 0.001), batches of 256, and 35 epochs
of 16,384 samples. Sampling is balanced with replacement, so an epoch is
an optimization budget rather than one pass over every recording. Training
uses log-amplitude/feature perturbations and 4% zero-feature silence.
There are 25 floating-point warmup epochs and ten Q10 fake-quantized epochs;
only checkpoints from the last ten epochs are eligible for export.
Hidden spike activity has a 1e-4 regularizer. Full loss and validation
histories, configuration, device, and wall time are retained for all six runs.
Training used an NVIDIA RTX 4060 Laptop GPU with PyTorch 2.11.0+cu128.

The checkpoint and score threshold maximize validation F1. The unconstrained
validation winner was rate seed 2. The required 250 ms hop then excludes
rate encoding on measured processor latency, so the highest-validation-F1
current model is deployed. This feasibility decision uses latency and
validation F1, not held-out test accuracy. Test scores were not used for
training, threshold calibration, or seed selection.

## Frontend, network, and fixed-point semantics

The PC pads or truncates mono PCM16 audio to one second at 16 kHz. It uses
a 400-sample periodic Hann, 160-sample hop, 512-point FFT, 24 normalized
triangular HTK mel filters from 80 to 7600 Hz, log-power clipping to
[-80,0] dB, and mean pooling to 32 time bins. Values map to uint8 and
flatten mel-major to 768 bytes. Training and the PC demo call the same
frontend; the test suite checks cached features against WAV replay.

Both SNNs use 64 LIF neurons, 12 steps, beta=7/8, threshold one, immediate
subtractive reset, and an accumulated **nonspiking linear readout**. Only
the hidden layer emits spikes. Every inference resets membrane/phase state.
Current encoding caches W*x+b; rate encoding uses a deterministic per-pixel
phase accumulator (add input byte, spike at 255, subtract 255).

Weights use signed int16 Q10, biases/accumulators use int32. Constant drive
is trunc(sum(w*x)/255)+b. Membrane update is trunc(7*v/8)+drive; at v>=1024,
subtract 1024 and increment the spike count. Signed divisions truncate to
zero in both C and the independent NumPy oracle. The final margin threshold
is {chosen['decision_threshold']}. Export checks conservative accumulator bounds.
No softmax or floating-point operations run on the RISC-V processor.

## Detection results

| Encoding | Seed | Validation F1 | Test precision | Test recall | Test F1 | Negative clip FPR |
|---|---:|---:|---:|---:|---:|---:|
{table}

Across three seeds, current F1 is {np.mean(by['current']):.4f} ± {np.std(by['current'],ddof=1):.4f}
and rate F1 is {np.mean(by['rate']):.4f} ± {np.std(by['rate'],ddof=1):.4f}
(sample standard deviation, not a confidence interval). Three seeds are
too few for a strong significance claim.

The deployed confusion matrix is TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, TN={m['tn']}.
The quantized and unquantized selected checkpoint disagree on
{chosen['quantization_decision_disagreements']} test decisions at the same threshold;
that is quantization sensitivity, not a C/RTL implementation mismatch.
All {b['windows']} additional nonoverlapping background-noise windows had
{b['false_detections']} false detections. Background recordings were not
used in training; see `results/background_noise.json` for the six sources.

![Accuracy and simulated latency](results/comparison.png)

## Processor cost and memory

| Encoding | Mean cycles | Maximum cycles | Maximum ms at 100 MHz |
|---|---:|---:|---:|
| Current | {cur['cycles_mean']:.0f} | {cur['cycles_max']} | {cur['milliseconds_max_at_100mhz']:.2f} |
| Rate | {rate['cycles_mean']:.0f} | {rate['cycles_max']} | {rate['milliseconds_max_at_100mhz']:.2f} |

Mean latency improves by {rate['cycles_mean']/cur['cycles_mean']:.2f}x. These are actual
cycles of the compiled RV32IM binary executing on the repository RTL, not
PC timing or instruction-count estimates. The 40 vectors include 12 positive,
12 negative, eight margin-boundary clips, repeated inputs, zero/saturated
inputs, and four fixed-seed random inputs. Maxima are observed stress-test
values, not a formal worst-case-execution-time proof. Ethernet, PC feature
extraction, Linux scheduling, and one-second window accumulation are outside
the firmware timer interval; 59.82 ms is not end-to-end microphone latency.

The constant projection performs 49,152 MACs once per clip; rate encoding
scans 589,824 input/weight positions over 12 steps and adds active weights.
Both perform 768 LIF updates and a 128-term readout. Deployed mean hidden
spikes are {chosen['hidden_spikes_mean']:.2f} per test clip. The C readout multiplies
accumulated spike counts; it is not an event-dispatched hardware readout.
Spike counts and operation counts are cost proxies only. No energy was measured.

The model uses {chosen['parameter_bytes']:,} bytes, with 768 input bytes and
a reserved 16 KiB stack. Code is below 0x10000; mailbox is at 0x10400;
input at 0x10800; model at 0x18000–0x3bfff; stack at 0x3c000–0x3ffff.
The linker rejects code/model overlap. The flat binary includes address gaps;
file length is therefore larger than parameter storage.

## Verification and FPGA implementation

1. The shared frontend, hand-computed LIF case, signed arithmetic, threshold
   ties, TCP framing, malformed lengths, and real-model TCP replay pass pytest.
2. The production inference C code, compiled natively, matches every score
   and hidden-spike count for all **11,005** held-out clips.
3. The same C compiled for RV32IM matches the independent oracle on all
   **40** RTL vectors for each encoding. Firmware is loaded through the
   host BRAM port and fetched/executed by the real PicoRV32 RTL. CPU traps
   and mailbox timeouts fail the harness.
4. Icarus four-state tests cover independent AXI AW/W arrival, backpressure,
   byte enables, read/write overlap, LED retrigger/cancel/reset, initialized
   timer and wraparound. Verilator also executes and measures one full
   **100,000,000-cycle** LED pulse. Duplicate requests and malformed commands
   are checked before/after normal firmware inference.
5. Vivado {h['tool'].split()[-1]} implemented XC7Z020 at 100 MHz: setup slack
   +{h['setup_slack_ns']:.3f} ns, hold slack +{h['hold_slack_ns']:.3f} ns;
   {h['resources']['luts']} LUTs, {h['resources']['registers']} registers,
   {h['resources']['bram36']} BRAM36 tiles, {h['resources']['dsps']} DSPs.
   Bitstream generation succeeded; routed implementation reported no
   critical warnings or errors. DRC retains ten advisory DSP-pipelining
   warnings. The ten LED outputs intentionally have no external synchronous
   output-delay requirement. Internal endpoints are constrained.

Hardware fixes include AXI arbitration across a pending BRAM read, byte
strobes on syscon scratch, defined timer initialization, Zynq DDR/fixed-I/O
connections, and the CFGBVS constraint spelling. The added LED countdown
runs independently of firmware and retriggers for one second on each positive
request. ABI version 2 prevents use of the old bitstream with the new server.

The raw timing/utilization/DRC reports and artifact SHA256 values are retained.
The vendored board preset still produces PS DDR skew advisories during IP
configuration; its board trace parameters were not arbitrarily altered.

## Demonstration and remaining physical work

The PC microphone frontend sends bounded, versioned TCP requests to an ARM
server. The server loads the firmware/model into PL BRAM, posts a mailbox
request, and returns the core's scores, cycle count, spikes, and decision.
This can also run locally with an integer-model server; simulated-server
cycle fields are zero to avoid presenting fabricated processor timing.

No board was available. Ethernet behavior on the PYNQ, Linux/PS clock setup,
physical LED timing, actual microphone accuracy, continuous-speech false
accepts per hour, acoustic robustness, power, and long-duration stability
remain physical acceptance checks. The isolated-word recall is about 85%,
so missed detections remain a substantive model limitation. A 1 s sliding
window does not constitute a validated wake-word product. Do not claim board
measurements or energy savings from these results.

## Reproducibility and references

`README.md` contains exact environment, training, export, simulation, and
future-board commands. `EXPERIMENT_PLAN.md` states the controlled comparison.
`deploy/` contains the selected model, firmware, bitstream, and hash manifest;
`results/` contains all six integer models, training histories, metrics, test
vectors, and implementation reports. Dataset audio and floating checkpoints
are local ignored artifacts. Code was developed with AI assistance; all
reported measurements come from the retained execution results.

- Pete Warden, [Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition](https://arxiv.org/abs/1804.03209), 2018.
- [Official Speech Commands v0.02 archive](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz).
- [PicoRV32 repository](https://github.com/YosysHQ/picorv32); the core used here is the source vendored in this project.
- Course kickoff: `PresentationInstruction/main.pdf` (Group 2 requirements).
'''
    (ROOT/'REPORT.md').write_text(report,encoding='utf-8')
    slides=[
        ('Spiking keyword detection on PicoRV32','Group 2 | Keyword: yes\nSpeech Commands v0.02 | PYNQ-Z2\n\nGPU training, integer deployment, RTL verification\nBoard acceptance remains pending'),
        ('Question and hypothesis','Can a small SNN fit 256 KiB BRAM and finish within a 250 ms hop?\n\nHypothesis: caching the constant projection cuts latency without a large F1 penalty.\n\nCompare current and deterministic rate encoding under equal training budgets.'),
        ('Controlled experiment','Official speaker-disjoint splits: 84,843 / 9,981 / 11,005\nYes vs all other 34 words\n768 inputs -> 64 LIF neurons -> 2 linear readouts\n12 steps, beta 7/8, immediate subtractive reset\n3 seeds x 2 encodings x 35 epochs; final 10 use QAT\nCheckpoint and threshold selection use validation data only.'),
        ('Training and deployment share a frontend','16 kHz, one-second mono audio\n400-sample Hann / 160 hop / 512 FFT\n24 mel bands x 32 time bins -> 768 uint8 bytes\nSigned Q10 int16 weights and int32 accumulators\nIndependent NumPy, native C, and RV32IM implementations\nNo floating point on the RISC-V processor.'),
        ('Detection results',f'Deployment: {e["selected_run"]}\nPrecision {100*m["precision"]:.2f}% | Recall {100*m["recall"]:.2f}% | F1 {100*m["f1"]:.2f}%\nTP {m["tp"]}, FP {m["fp"]}, FN {m["fn"]}, TN {m["tn"]}\nBackground probes: 0 / 398 detections\n\nClip metrics do not establish continuous-speech false accepts/hour.'),
        ('Latency determines the demo model',f'Current: max {cur["milliseconds_max_at_100mhz"]:.2f} ms\nRate: max {rate["milliseconds_max_at_100mhz"]:.2f} ms\nMean speedup: {rate["cycles_mean"]/cur["cycles_mean"]:.2f}x\nRate wins validation F1 but misses the 250 ms hop.\nSelect the feasible current model by validation F1.\nTiming excludes audio-window accumulation, PC frontend and Ethernet.'),
        ('Measured comparison',None),
        ('Verification evidence','11,005 clips: native production C matches integer oracle exactly\n40 vectors per encoding: actual PicoRV32 execution matches exactly\nTraps, timeouts, malformed requests, and duplicate sequence checks\nAXI backpressure, byte lanes, independent AW/W, read/write contention\n100,000,000-cycle LED pulse; retrigger/reset/cancel/wrap checks\nReal-model local TCP replay and shared WAV frontend tests'),
        ('FPGA implementation',f'Vivado 2025.2 | XC7Z020 | 100 MHz\nSetup slack +{h["setup_slack_ns"]:.3f} ns; hold slack +{h["hold_slack_ns"]:.3f} ns\n{h["resources"]["luts"]} LUTs; {h["resources"]["registers"]} registers\n{h["resources"]["bram36"]} BRAM36 tiles; {h["resources"]["dsps"]} DSPs\n98,824-byte integer model; 16 KiB reserved stack\nBitstream generated; no implementation errors or critical warnings.'),
        ('Demo and limitations','PC microphone -> mel features -> TCP -> ARM mailbox -> PicoRV32\nPositive inference retriggers hardware LED0 for one second.\n\nNot yet measured: physical board, microphone, Ethernet latency, power.\nAbout 15% of isolated yes clips are missed.\nNo general energy or SNN-vs-ANN claim.\nArtifacts, hashes, raw reports, and reproduction commands are included.')]
    with PdfPages(ROOT/'presentation.pdf') as pdf:
        for index,(title,body) in enumerate(slides):
            if body is None:
                pdf.savefig(fig)
                continue
            page=plt.figure(figsize=(13.33,7.5),facecolor='#f4f7fa')
            page.text(.07,.85,title,fontsize=28,weight='bold',color='#103448')
            wrapped='\n\n'.join(textwrap.fill(line,85) if line else '' for line in body.splitlines())
            page.text(.07,.72,wrapped,fontsize=18,color='#243e4a',va='top',linespacing=1.1)
            page.text(.07,.045,f'5XPG0 | Group 2 | Simulation and implementation evidence    {index+1}/{len(slides)}',fontsize=10,color='#526e7e')
            pdf.savefig(page);plt.close(page)
    plt.close(fig)


if __name__=='__main__': main()
