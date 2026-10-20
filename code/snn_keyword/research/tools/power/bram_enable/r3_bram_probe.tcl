# Diagnostic (R3 2E): where does the enable-independent BRAM power of the idle state come from?
#   vivado -mode batch -source states_r3/r3_bram_probe.tcl -tclargs reports_r3_rw1/routed.dcp states_r3/probe_rw1
set dcp [lindex $argv 0]
set out [file normalize [lindex $argv 1]]
file mkdir $out
open_checkpoint $dcp
set_operating_conditions -ambient_temp 25 -process typical
set gate [get_cells -hier -filter {REF_NAME =~ BUFG* && NAME =~ *u_cpu_gate*}]
set gclk_o [get_pins -of_objects $gate -filter {REF_PIN_NAME == O}]
set mem [get_cells -hier -filter {REF_NAME =~ RAMB* && NAME =~ u_soc/mem_reg*}]
set r0 [lindex [lsort $mem] 0]
set r1 [lindex [lsort $mem] 1]
proc bram_line {f tag} {
    set fp [open $f r]; set t [read $fp]; close $fp
    foreach l [split $t \n] { if {[string match "| Block RAM*" $l] || [string match "| Signals*" $l] || [string match "| Clocks*" $l]} { puts "PROBE $tag: $l" } }
}
proc idle_base {gclk_o mem} {
    reset_switching_activity -all
    set_switching_activity -default_toggle_rate 12.5 -default_static_probability 0.5
    create_clock -name cpu_gclk -period 1e10 $gclk_o
    foreach pat {ENARDEN ENBWREN WEA* WEBWE*} {
        set_switching_activity -static_probability 0 -toggle_rate 0 [get_pins -of $mem -filter "REF_PIN_NAME =~ $pat"]
    }
}
# 1. the idle state as in r3_power_states.tcl, and the activity Vivado then uses on each pin kind of two blocks
idle_base $gclk_o $mem
set pins {}
foreach r [list $r0 $r1] {
    foreach p {CLKARDCLK CLKBWRCLK ENARDEN ENBWREN REGCEAREGCE REGCEB RSTRAMARSTRAM RSTRAMB RSTREGARSTREG RSTREGB WEA[0] WEBWE[0] ADDRARDADDR[0] ADDRARDADDR[15] ADDRBWRADDR[0] ADDRBWRADDR[15] DIADI[0] DIBDI[0] CASCADEINA CASCADEINB} {
        set pin [get_pins -quiet $r/$p]
        if {[llength $pin]} { lappend pins $pin }
    }
}
report_switching_activity -file $out/activity_idle.txt $pins
set fh [open $out/drivers.txt w]
foreach r [list $r0 $r1] {
    puts $fh "$r RAM_EXTENSION_A=[get_property RAM_EXTENSION_A $r] RAM_MODE=[get_property RAM_MODE $r] WRITE_MODE_A=[get_property WRITE_MODE_A $r] WRITE_MODE_B=[get_property WRITE_MODE_B $r] DOA_REG=[get_property DOA_REG $r] DOB_REG=[get_property DOB_REG $r]"
    foreach p {ENARDEN ENBWREN ADDRARDADDR[15] ADDRBWRADDR[15] REGCEAREGCE REGCEB} {
        set net [get_nets -quiet -of [get_pins $r/$p]]
        set drv [get_pins -quiet -leaf -of $net -filter {DIRECTION == OUT}]
        puts $fh "  $p <- net $net driver [get_property REF_NAME [get_cells -quiet -of $drv]] $drv"
    }
}
close $fh
report_power -file $out/p1_idle.rpt
bram_line $out/p1_idle.rpt idle
# 2. idle + every other BRAM data/address input held still (a sleeping CPU and an idle PS do not change them)
idle_base $gclk_o $mem
foreach pat {ADDRARDADDR* ADDRBWRADDR* DIADI* DIBDI* DIPADIP* DIPBDIP*} {
    set_switching_activity -static_probability 0 -toggle_rate 0 [get_pins -of $mem -filter "REF_PIN_NAME =~ $pat"]
}
report_power -file $out/p2_idle_inputs_still.rpt
bram_line $out/p2_idle_inputs_still.rpt idle+inputs_still
# 3. as 2, plus the output-register enables and resets held at 0
idle_base $gclk_o $mem
foreach pat {ADDRARDADDR* ADDRBWRADDR* DIADI* DIBDI* DIPADIP* DIPBDIP* REGCE* RST*} {
    set_switching_activity -static_probability 0 -toggle_rate 0 [get_pins -quiet -of $mem -filter "REF_PIN_NAME =~ $pat"]
}
report_power -file $out/p3_idle_all_still.rpt
bram_line $out/p3_idle_all_still.rpt idle+all_still
# 4. both ports enabled every cycle with the inputs still (enable cost alone)
idle_base $gclk_o $mem
foreach pat {ADDRARDADDR* ADDRBWRADDR* DIADI* DIBDI* DIPADIP* DIPBDIP*} {
    set_switching_activity -static_probability 0 -toggle_rate 0 [get_pins -of $mem -filter "REF_PIN_NAME =~ $pat"]
}
foreach pat {ENARDEN ENBWREN} {
    set_switching_activity -static_probability 1 -toggle_rate 0 [get_pins -of $mem -filter "REF_PIN_NAME =~ $pat"]
}
report_power -file $out/p4_en1_inputs_still.rpt
bram_line $out/p4_en1_inputs_still.rpt en1+inputs_still
close_design
