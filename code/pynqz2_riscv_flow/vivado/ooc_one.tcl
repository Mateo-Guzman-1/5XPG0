# ooc_one.tcl -- SKETCH (not run here: no Vivado in my sandbox). Out-of-context
# synthesis of ONE snn_layer configuration, to measure area vs. parallelism P
# without waiting for the full ~15 min SoC build.
#
# Run from vivado/ (same place as build.tcl), Vivado 2024.1:
#   ./vivado2024.sh -mode batch -source ooc_one.tcl -tclargs <P> <XBITS> [impl]
#
# Sweep example:
#   for x in 1 8; do for p in 4 8 16 32 64; do
#       ./vivado2024.sh -mode batch -source ooc_one.tcl -tclargs $p $x
#   done; done
#
# One CSV line per run is appended to ooc_sweep.csv.
#   - default: post-SYNTHESIS numbers (WNS is optimistic: no routing delays)
#   - "impl" : also opt/place/route (still out-of-context) -> credible WNS
#
# One run per Vivado session on purpose: robust, and easy to parallelise.

set P  [lindex $argv 0]
set XB [lindex $argv 1]
set impl [expr {[llength $argv] > 2 && [lindex $argv 2] eq "impl"}]

set part xc7z020clg400-1                       ;# same part as build.tcl
set rtl  [file normalize ../rtl_sketches/snn_layer.v]
set tag  P${P}_X${XB}
file mkdir ooc_reports

read_verilog $rtl
synth_design -top snn_layer -part $part -mode out_of_context \
             -generic P=$P -generic XBITS=$XB

create_clock -period 10.0 -name clk [get_ports clk]     ;# FCLK0 = 100 MHz in this flow

if {$impl} {
    opt_design
    place_design
    route_design
}

report_utilization    -file ooc_reports/util_$tag.rpt
report_timing_summary -file ooc_reports/timing_$tag.rpt

# ---- pull a few numbers out of the utilization report (adjust if the format differs)
proc util_row {rpt key} {
    set fp [open $rpt r]
    set data [read $fp]
    close $fp
    foreach line [split $data "\n"] {
        if {[string first "| $key" $line] == 0} {
            return [string trim [lindex [split $line "|"] 2]]
        }
    }
    return NA
}
set lut  [util_row ooc_reports/util_$tag.rpt "Slice LUTs"]
set ff   [util_row ooc_reports/util_$tag.rpt "Slice Registers"]
set bram [util_row ooc_reports/util_$tag.rpt "Block RAM Tile"]
set dsp  [util_row ooc_reports/util_$tag.rpt "DSPs"]
set wns  [get_property SLACK [get_timing_paths -max_paths 1 -setup]]

if {![file exists ooc_sweep.csv]} {
    set fh [open ooc_sweep.csv w]
    puts $fh "P,XBITS,impl,LUT,FF,BRAM_tiles,DSP,WNS_ns"
    close $fh
}
set fh [open ooc_sweep.csv a]
puts $fh "$P,$XB,$impl,$lut,$ff,$bram,$dsp,$wns"
close $fh

puts "OOC $tag -> LUT=$lut FF=$ff BRAM=$bram DSP=$dsp WNS=$wns ns"
exit
