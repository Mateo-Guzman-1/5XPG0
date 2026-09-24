/* linker.ld — firmware memory layout.
 *
 * One 256 KB BRAM, everything execute-in-place:
 *
 *   0x00000 .. 0x0FFFF  code + rodata + data + bss (this script, 64 KB max)
 *   0x10000             console ring      (1 KB, fixed address)
 *   0x10400             mailbox           (1 KB, fixed address)
 *   0x11000 ..          spike log ring     (fixed address)
 *   0x3D000 .. 0x3FFFF  stack (grows down from 0x40000)
 *
 * The fixed regions are NOT covered by any section here — the firmware
 * reaches them with absolute addresses from board.h.
 */

OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
    RAM (rwx) : ORIGIN = 0x00000000, LENGTH = 64K
}

SECTIONS
{
    .text :
    {
        KEEP(*(.text.init))
        *(.text .text.*)
        *(.rodata .rodata.*)
        *(.srodata .srodata.*)
    } > RAM

    .data :
    {
        __data_start = .;
        *(.data .data.*)
        *(.sdata .sdata.*)
        . = ALIGN(8);
        __data_end = .;
    } > RAM

    .bss (NOLOAD) :
    {
        . = ALIGN(8);
        __bss_start = .;
        *(.bss .bss.*)
        *(.sbss .sbss.*)
        *(COMMON)
        . = ALIGN(8);
        __bss_end = .;
    } > RAM

    __image_end = .;
    __stack_top = 0x40000;

    ASSERT(__image_end <= 0x10000 - 0x100, "firmware image overlaps the console ring at 0x10000")
}
