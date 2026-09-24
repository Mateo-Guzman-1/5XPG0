// start.S — crt0 for the AURA firmware (PicoRV32, RV32IM).
// The CPU fetches from address 0 after reset; this code must be first.

    .section .text.init, "ax", @progbits
    .globl _start
_start:
    la      sp, __stack_top
    la      a0, __bss_start
    la      a1, __bss_end
1:  bgeu    a0, a1, 2f
    sw      zero, 0(a0)
    addi    a0, a0, 4
    j       1b
2:  call    main
3:  j       3b
