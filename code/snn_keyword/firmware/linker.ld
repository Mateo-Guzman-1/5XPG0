OUTPUT_ARCH(riscv)
ENTRY(_start)
MEMORY {
    CODE (rwx) : ORIGIN = 0, LENGTH = 64K
    MODEL (r) : ORIGIN = 0x18000, LENGTH = 144K
}
SECTIONS {
    .text : { KEEP(*(.text.init)) *(.text .text.*) *(.rodata .rodata.*) *(.srodata .srodata.*) } > CODE
    .data : { *(.data .data.*) *(.sdata .sdata.*) } > CODE
    .bss (NOLOAD) : {
        . = ALIGN(8); __bss_start = .;
        *(.bss .bss.*) *(.sbss .sbss.*) *(COMMON)
        . = ALIGN(8); __bss_end = .;
    } > CODE
    .model : { *(.model) } > MODEL
    __stack_top = 0x40000;
    ASSERT(__bss_end < 0x10000, "Code overlaps mailbox")
    ASSERT(SIZEOF(.model) <= 144K, "Model overlaps 16 KiB stack")
}
