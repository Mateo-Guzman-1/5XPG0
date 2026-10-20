# Diagnostic 2 (R3 2E): is the enable-independent BRAM power the BRAM clock? Idle state (enables 0, inputs still),
# then the BRAM clock pins declared as a very slow clock: CPU port pins only (CLKARDCLK), then both ports.
set dcp [lindex $argv 0]
set out [file normalize [lindex $argv 1]]
file mkdir $out
open_checkpoint $dcp
set_operating_conditions -ambient_temp 25 -process typical
set gate [get_cells -hier -filter {REF_NAME =~ BUFG* && NAME =~ *u_cpu_gate*}]
set gclk_o [get_pins -of_objects $gate -filter {REF_PIN_NAME == O}]
set mem [get_cells -hier -filter {REF_NAME =~ RAMB* && NAME =~ u_soc/mem_reg*}]
proc bram_line {f tag} {
    set fp [open $f r]; set t [read $fp]; close $fp
    foreach l [split $t \n] { if {[string match "| Block RAM*" $l] || [string match "| Clocks*" $l]} { puts "PROBE2 $tag: $l" } }
}
proc idle_still {gclk_o mem} {
    reset_switching_activity -all
    set_switching_activity -default_toggle_rate 12.5 -default_static_probability 0.5
    create_clock -name cpu_gclk -period 1e10 $gclk_o
    foreach pat {ENARDEN ENBWREN WEA* WEBWE* ADDRARDADDR* ADDRBWRADDR* DIADI* DIBDI* DIPADIP* DIPBDIP*} {
        set_switching_activity -static_probability 0 -toggle_rate 0 [get_pins -of $mem -filter "REF_PIN_NAME =~ $pat"]
    }
}
set fh [open $out/clock_pins.txt w]
foreach p {CLKARDCLK CLKBWRCLK} {
    set pin [get_pins [lindex [lsort $mem] 0]/$p]
    puts $fh "$p: net [get_nets -of $pin], clocks [get_clocks -quiet -of_objects $pin]"
}
close $fh
idle_still $gclk_o $mem
report_power -file $out/q0_idle_still.rpt
bram_line $out/q0_idle_still.rpt idle_still
idle_still $gclk_o $mem
create_clock -name bram_a_slow -period 1e10 [get_pins -of $mem -filter {REF_PIN_NAME == CLKARDCLK}]
report_power -file $out/q1_cpu_port_clock_off.rpt
bram_line $out/q1_cpu_port_clock_off.rpt cpu_port_clock_off
idle_still $gclk_o $mem
create_clock -name bram_a_slow -period 1e10 [get_pins -of $mem -filter {REF_PIN_NAME == CLKARDCLK}]
create_clock -name bram_b_slow -period 1e10 [get_pins -of $mem -filter {REF_PIN_NAME == CLKBWRCLK}]
report_power -file $out/q2_both_port_clocks_off.rpt
bram_line $out/q2_both_port_clocks_off.rpt both_port_clocks_off
# enables 1 on the CPU port with its clock slow: does the enable cost need the clock?
idle_still $gclk_o $mem
set_switching_activity -static_probability 1 -toggle_rate 0 [get_pins -of $mem -filter {REF_PIN_NAME == ENARDEN}]
report_power -file $out/q3_cpu_en1.rpt
bram_line $out/q3_cpu_en1.rpt cpu_en1_clock_on
close_design
