# r3_vivado.tcl - R3 build (research/PLAN_R3_GATING.md 2D): build.tcl with top-level generics, then checks and reports.
#
# Run it from the vivado/ folder of a `git archive` copy of code/pynqz2_riscv_flow (never the tracked tree), with
# this file copied next to build.tcl:
#   vivado -mode batch -nojournal -source r3_vivado.tcl -tclargs <projdir> <out.bit> <reportdir> <POWER_OPT> <READ_WAIT>
# The logic of vivado_power.tcl (the same board-path workaround), plus:
#   - after build.tcl's `set_property top` line: set_property generic {POWER_OPT=<p> READ_WAIT=<r>} on sources_1;
#   - check: exactly one BUFG* cell under u_soc/u_cpu_gate when POWER_OPT = 1 (none when 0), otherwise the clock
#     gate became a latch on the clock (stop);
#   - reports: timing summary, the BUFGCE CE path, utilization, DRC, clock buffers, routed.dcp, and the
#     vectorless power reports at 12.5 / 6.25 / 25 % (all clocks running; the per-state reports come from
#     r3_power_states.tcl on routed.dcp).

if {$argc < 5} { error "usage: -tclargs <projdir> <out.bit> <reportdir> <POWER_OPT> <READ_WAIT>" }
set report_dir [file normalize [lindex $argv 2]]
set power_opt [lindex $argv 3]
set read_wait [lindex $argv 4]

set fp [open build.tcl r]
set build [read $fp]
close $fp
set repo [file normalize ./board_files]
if {[string first " " $repo] >= 0} {
    set copy [file join [file normalize $::env(TEMP)] 5xpg0_board_files_[file tail [lindex $argv 0]]]
    file delete -force $copy
    file copy -force $repo $copy
    set repo $copy
}
set line {set_param board.repoPaths [list [file normalize ./board_files]]}
if {[string first $line $build] < 0} { error "build.tcl changed: board.repoPaths line not found" }
set build [string map [list $line "set_param board.repoPaths \[list {$repo}\]"] $build]
set top {set_property top spike_top [current_fileset]}
if {[string first $top $build] < 0} { error "build.tcl changed: set_property top line not found" }
set build [string map [list $top "$top\nset_property generic {POWER_OPT=$power_opt READ_WAIT=$read_wait} \[current_fileset\]"] $build]
puts "board.repoPaths: $repo"
puts "R3 GENERICS: POWER_OPT=$power_opt READ_WAIT=$read_wait"

eval $build   ;# synthesis, implementation, bitstream; leaves the routed impl_1 design open

file mkdir $report_dir
foreach f {timing_summary.rpt utilization.rpt drc.rpt} {
    file copy -force $projdir/$f $report_dir/$f
}

# Clock buffers: the CPU clock gate must be exactly one BUFGCE (a BUFGCTRL primitive) when POWER_OPT = 1.
set fh [open $report_dir/clock_buffers.txt w]
set bufs [get_cells -hier -filter {REF_NAME =~ BUFG*}]
foreach b $bufs { puts $fh "$b [get_property REF_NAME $b]" }
set gate [get_cells -quiet -hier -filter {REF_NAME =~ BUFG* && NAME =~ *u_cpu_gate*}]
puts $fh "u_cpu_gate buffers: [llength $gate]"
close $fh
set want [expr {$power_opt != 0 ? 1 : 0}]
if {[llength $gate] != $want} { error "R3 CHECK FAILED: [llength $gate] BUFG* cells under u_cpu_gate, expected $want" }
puts "R3 CHECK: u_cpu_gate buffers [llength $gate] (expected $want); all clock buffers [llength $bufs]"

if {$want} {
    set ce [get_pins -of_objects $gate -filter {REF_PIN_NAME =~ CE*}]
    report_timing -through $ce -max_paths 5 -file $report_dir/timing_ce.rpt
}
report_timing -max_paths 10 -file $report_dir/timing_paths.rpt
write_checkpoint -force $report_dir/routed.dcp

set_operating_conditions -ambient_temp 25 -process typical
report_power -hierarchical_depth 4 -file $report_dir/power.rpt
foreach rate {6.25 25} {
    set_switching_activity -default_toggle_rate $rate
    report_power -hierarchical_depth 4 -file $report_dir/power_toggle$rate.rpt
}
set_switching_activity -default_toggle_rate 12.5
puts "R3 REPORTS: $report_dir"
