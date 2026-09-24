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
targets -set -filter {name =~ "ARM*#0"}
catch {stop}
source $root/deploy/ps7_init.tcl
ps7_init
targets -set -filter {name =~ "xc7z020"}
fpga -file $bit
targets -set -filter {name =~ "ARM*#0"}
ps7_post_config
configparams force-mem-access 1

proc rd {address} {return [expr {wide([lindex [mrd -force -value $address] 0])}]}
if {[rd 0x40040014] != 0x534b454c} {error "Spike SoC magic not found: wrong bitstream?"}
mwr -force 0x40040000 1
mwr -force -bin -file $bin 0x40000000 [expr {[file size $bin] / 4}]
mwr -force 0x40010400 0 16
mwr -force 0x40040000 0
after 300
if {[rd 0x40010430] != 0x4b575331} {error "Firmware did not start (mailbox magic missing)"}
puts [format "READY abi=0x%08x threshold=%d input_bytes=%d stream_threshold=%d" \
    [rd 0x4004001c] [rd 0x40010434] [rd 0x40010438] [rd 0x4001043c]]
exit
