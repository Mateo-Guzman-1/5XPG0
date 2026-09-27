# JTAG bring-up WITHOUT PYNQ Linux (workaround, see README "JTAG workaround").
# Usage (from any directory):
#   xsdb jtag/bringup.tcl [bitstream] [firmware]
# Defaults: deploy/keyword_kdot.bit and deploy/keyword_kdot.bin.
#
# Replaces what the FSBL + PYNQ normally do: ps7_init sets up the PS clocks
# (FCLK0 = 100 MHz), the PL is programmed, ps7_post_config enables the PS-PL
# level shifters, then the PicoRV32 is held in reset while the firmware image
# is written into BRAM through the PS AXI GP0 port, and released.
set root [file normalize [file join [file dirname [info script]] ..]]
set bit [file normalize [expr {[llength $argv] > 0 ? [lindex $argv 0] : "$root/deploy/keyword_kdot.bit"}]]
set bin [file normalize [expr {[llength $argv] > 1 ? [lindex $argv 1] : "$root/deploy/keyword_kdot.bin"}]]

connect -url tcp:127.0.0.1:3121
# If PYNQ Linux booted, the PS already runs FCLK0 at 100 MHz (IO PLL / 5 / 2)
# with the level shifters on: only the PL is reprogrammed, Linux keeps running,
# and memory is accessed physically through the APU debug port.
targets -set -filter {name =~ "APU"}
proc apu_rd {address} {return [expr {wide([lindex [mrd -force -value $address] 0])}]}
if {[apu_rd 0xF8000170] == 0x00200500 && [apu_rd 0xF8000900] == 0xf} {
    puts "PS already configured (PYNQ Linux running): programming the PL only"
    targets -set -filter {name =~ "xc7z020"}
    fpga -file $bit
    targets -set -filter {name =~ "APU"}
} else {
    targets -set -filter {name =~ "ARM*#0"}
    catch {stop}
    source $root/deploy/ps7_init.tcl
    ps7_init
    targets -set -filter {name =~ "xc7z020"}
    fpga -file $bit
    targets -set -filter {name =~ "ARM*#0"}
    ps7_post_config
    configparams force-mem-access 1
}

proc rd {address} {return [expr {wide([lindex [mrd -force -value $address] 0])}]}
if {[rd 0x40040014] != 0x534b454c} {error "Spike SoC magic not found: wrong bitstream?"}
mwr -force 0x40040000 1
mwr -force -bin -file $bin 0x40000000 [expr {[file size $bin] / 4}]
mwr -force 0x40010400 0 16
mwr -force 0x40040000 0
after 300
set magic [rd 0x40010430]
if {$magic == 0x4b575331} {
    puts [format "READY abi=0x%08x firmware=KWS1 threshold=%d input_bytes=%d stream_threshold=%d" \
        [rd 0x4004001c] [rd 0x40010434] [rd 0x40010438] [rd 0x4001043c]]
} elseif {$magic == 0x4b575333} {
    puts [format "READY abi=0x%08x firmware=KWS3 threshold=%d frame_bytes=%d max_frames=%d" \
        [rd 0x4004001c] [rd 0x40010434] [rd 0x40010438] [rd 0x4001043c]]
} else {error "Firmware did not start (mailbox magic missing)"}
exit
