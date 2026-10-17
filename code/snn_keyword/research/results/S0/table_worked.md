| Fabric | Step | Value |
|---|---|---|
| RV32IM | Mean / max cycles (board) | 8,872,912 / 8,877,865 |
|  | E_inf = P_PL,dyn x C_mean / f = 0.278 W x 88.729 ms | 24,666.7 uJ |
|  | E_inf,all = (P_PL,dyn + P_PL,stat) x C_mean / f | 32,208.7 uJ |
|  | (a) busy-poll: (P_PL,dyn + P_PL,stat) x T = 0.363 W x 0.25 s | 90.75 mJ = 363.0 mW |
|  | (b) gated: (P_PL,stat + P_infra,dyn) x T + P_core,dyn x C_mean / f | 47.21 mJ = 188.8 mW |
|  | (c) scaled: f_min = 1.1 C_max / T = 39.06 MHz; P_PL,stat x T + P_PL,dyn x 1.1 C_max / f | 48.40 mJ = 193.6 mW |
|  | Energy per synaptic multiply-add (73,150 per inference) | 337.2 nJ |
| RV32IM + kdot | Mean / max cycles (board) | 390,992 / 395,945 |
|  | E_inf = P_PL,dyn x C_mean / f = 0.282 W x 3.910 ms | 1,102.6 uJ |
|  | E_inf,all = (P_PL,dyn + P_PL,stat) x C_mean / f | 1,434.9 uJ |
|  | (a) busy-poll: (P_PL,dyn + P_PL,stat) x T = 0.367 W x 0.25 s | 91.75 mJ = 367.0 mW |
|  | (b) gated: (P_PL,stat + P_infra,dyn) x T + P_core,dyn x C_mean / f | 24.32 mJ = 97.3 mW |
|  | (c) scaled: f_min = 1.1 C_max / T = 1.74 MHz; P_PL,stat x T + P_PL,dyn x 1.1 C_max / f | 22.48 mJ = 89.9 mW |
|  | Energy per synaptic multiply-add (73,150 per inference) | 15.1 nJ |
