| Quantity (W) | RV32IM | RV32IM + kdot |
|---|---|---|
| Total on-chip power | 1.683 | 1.688 |
| Dynamic power | 1.536 | 1.540 |
| Device static power | 0.147 | 0.147 |
| Dynamic: PS7 | 1.258 | 1.258 |
| Dynamic: Clocks | 0.011 | 0.011 |
| Dynamic: Slice Logic | 0.005 | 0.006 |
| Dynamic: Signals | 0.008 | 0.009 |
| Dynamic: Block RAM | 0.251 | 0.251 |
| Dynamic: DSPs | 0.003 | 0.006 |
| Dynamic: I/O | 0.000 | 0.000 |
| Dynamic: core u_soc (CPU, kdot, BRAM) | 0.270 | 0.274 |
|   of which PicoRV32 u_cpu | 0.011 | 0.011 |
|   of which kdot u_kdot | -- | 0.005 |
| Dynamic: AXI interface u_ps_if | 0.002 | 0.002 |
| Dynamic: AXI protocol converter pc0 | 0.005 | 0.005 |
| Static, PL rails (Vccint, Vccaux, Vccbram, Vccadc, Vcco) | 0.085 | 0.085 |
| Static, PS rails (Vccp*, DDR, MIO) | 0.062 | 0.062 |
| Derived: PL dynamic P_PL,dyn = dynamic - PS7 | 0.278 | 0.282 |
| Derived: core P_core,dyn (u_soc) | 0.270 | 0.274 |
| Derived: interface P_infra,dyn = P_PL,dyn - P_core,dyn | 0.008 | 0.008 |
