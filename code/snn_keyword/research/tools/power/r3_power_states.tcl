# r3_power_states.tcl - per-state Vivado power of a routed R3 checkpoint (research/PLAN_R3_GATING.md 2E).
#
#   vivado -mode batch -nojournal -source r3_power_states.tcl -tclargs <routed.dcp> <outdir> <duties.tcl>
#
# Ported from the streaming design's cpu-sleep_power.tcl (gated clock) and bram_power.tcl (BRAM enables).
# Common settings as in S0: 25 C, typical process, default toggle rate 12.5 % (also 6.25 and 25 %), default
# static probability 0.5.
# - report_power ignores a BUFGCE's CE (Power 33-324) and clocks the gated domain at 100 MHz. Each state therefore
#   gives the gated CPU clock its average rate: create_clock on the BUFGCE output with period 10 ns / duty.
# - BRAM enables: each block's EN pin gets the static probability given in duties.tcl (per block: 1 for the
#   width-sliced memory as built, whose enables synthesis ties to 1; 1/64 for the address-decoded ram_decomp layout,
#   one block per port per cycle) and toggle rate min(99, 199 min(p, 1 - p)) %; the WE pins likewise with the
#   simulated write duty. The physical port driven from the PS read/write logic (fan-in from u_ps_if) is the PS port.
#   (G2: an earlier version gave EN the simulated read duties; the built netlist has no such enables.)
# duties.tcl (written by r3_verify.py duties) sets:
#   set states { <name> <cpu_clock_duty> <pb_en> <pb_we> <pa_en> <pa_we> ... }
# Writes <outdir>/power_<name>_tr<rate>.rpt, clocks_<name>.rpt and bram_ports.txt.

if {$argc < 3} { error "usage: -tclargs <routed.dcp> <outdir> <duties.tcl>" }
set dcp [lindex $argv 0]
set out [file normalize [lindex $argv 1]]
source [lindex $argv 2]
file mkdir $out
open_checkpoint $dcp
set_operating_conditions -ambient_temp 25 -process typical

set gate [get_cells -hier -filter {REF_NAME =~ BUFG* && NAME =~ *u_cpu_gate*}]
if {[llength $gate] != 1} { error "expected one BUFG* under u_cpu_gate, found [llength $gate]" }
set gclk_o [get_pins -of_objects $gate -filter {REF_PIN_NAME == O}]

set mem [get_cells -hier -filter {REF_NAME =~ RAMB* && NAME =~ u_soc/mem_reg*}]
set n [llength $mem]
set r0 [lindex [lsort $mem] 0]
# Which physical port is the PS's: the one whose enable has a start point in u_ps_if (rstate / write logic).
proc from_ps {pin} {
    set sp [all_fanin -flat -startpoints_only [get_pins $pin]]
    return [expr {[llength [lsearch -all -inline $sp *u_ps_if*]] > 0}]
}
set a_ps [from_ps $r0/ENARDEN]
set b_ps [from_ps $r0/ENBWREN]
if {$a_ps == $b_ps} { error "cannot tell the PS port: ENARDEN from PS $a_ps, ENBWREN from PS $b_ps" }
if {$a_ps} { set ps_pins {ENARDEN WEA*}; set cpu_pins {ENBWREN WEBWE*} } else { set ps_pins {ENBWREN WEBWE*}; set cpu_pins {ENARDEN WEA*} }
set fh [open $out/bram_ports.txt w]
puts $fh "CPU memory blocks (u_soc/mem_reg*): $n"
puts $fh "gated clock: $gate -> $gclk_o"
puts $fh "PS port pins: $ps_pins; CPU port pins: $cpu_pins (ENARDEN from PS: $a_ps, ENBWREN from PS: $b_ps)"
dict set kinds x 0
foreach r $mem {
    set k "[get_property REF_NAME $r] RA=[get_property READ_WIDTH_A $r] RB=[get_property READ_WIDTH_B $r] CASC=[get_property RAM_EXTENSION_A $r]"
    if {[dict exists $kinds $k]} { dict incr kinds $k } else { dict set kinds $k 1 }
}
dict unset kinds x
dict for {k v} $kinds { puts $fh "$v x $k" }
close $fh

proc pins_of {mem pat} { return [get_pins -of $mem -filter "REF_PIN_NAME =~ $pat"] }
foreach {name cpu pb_en pb_we pa_en pa_we} $states {
    foreach rate {12.5 6.25 25} {
        reset_switching_activity -all
        set_switching_activity -default_toggle_rate $rate -default_static_probability 0.5
        create_clock -name cpu_gclk -period [expr {10.0 / max($cpu, 1e-9)}] $gclk_o
        foreach {pins duty} [list [lindex $cpu_pins 0] $pb_en [lindex $cpu_pins 1] $pb_we \
                                  [lindex $ps_pins 0] $pa_en [lindex $ps_pins 1] $pa_we] {
            set p [expr {min(1.0, max(0.0, double($duty)))}]
            set t [expr {min(99.0, 199.0 * min($p, 1.0 - $p))}]
            set_switching_activity -static_probability $p -toggle_rate $t [pins_of $mem $pins]
        }
        report_power -hierarchical_depth 4 -file $out/power_${name}_tr$rate.rpt
        if {$rate == 12.5} { report_clocks -file $out/clocks_$name.rpt }
    }
    puts "R3 STATE $name: cpu clock duty $cpu, port B en $pb_en we $pb_we, port A en $pa_en we $pa_we"
}
close_design
