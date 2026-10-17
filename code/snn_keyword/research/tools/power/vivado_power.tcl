# vivado_power.tcl - build one fabric with build.tcl, then write Vivado's power reports.
#
# Run it from the vivado/ folder of a `git archive` copy of code/pynqz2_riscv_flow, never from the
# tracked tree (build.tcl's default outputs are tracked files). Copy this file next to build.tcl:
#   vivado -mode batch -nojournal -source vivado_power.tcl -tclargs <projdir> <out.bit> <reportdir>
# build.tcl reads the first two arguments; the third is the folder for the reports.
#
# The power estimate is vectorless: Vivado assumes a default activity (12.5 % toggle rate, static
# probability 0.5) for every net it cannot derive. Two extra reports at 6.25 % and 25 % show how
# much the absolute numbers depend on that assumption.

if {$argc < 3} { error "usage: -tclargs <projdir> <out.bit> <reportdir>" }
set report_dir [file normalize [lindex $argv 2]]

# build.tcl is run unchanged except for one tool-environment line. Vivado 2026.1 hands
# board.repoPaths to its child runs (out-of-context IP synthesis) without quotes, so a path that
# contains a space ("Semiconductor Technology") is split in two and the PYNQ-Z2 board files are not
# found. In that case the vendored board files are copied to a space-free folder under %TEMP%.
set fp [open build.tcl r]
set build [read $fp]
close $fp
set repo [file normalize ./board_files]
if {[string first " " $repo] >= 0} {
    # One copy per project, so that two builds can run at the same time.
    set copy [file join [file normalize $::env(TEMP)] 5xpg0_board_files_[file tail [lindex $argv 0]]]
    file delete -force $copy
    file copy -force $repo $copy
    set repo $copy
}
set line {set_param board.repoPaths [list [file normalize ./board_files]]}
if {[string first $line $build] < 0} { error "build.tcl changed: board.repoPaths line not found" }
set build [string map [list $line "set_param board.repoPaths \[list {$repo}\]"] $build]
puts "board.repoPaths: $repo"

eval $build   ;# synthesis, implementation, bitstream; leaves the routed impl_1 design open

file mkdir $report_dir
foreach f {timing_summary.rpt utilization.rpt drc.rpt} {
    file copy -force $projdir/$f $report_dir/$f
}

# Vivado's defaults, written out so that the report states them.
set_operating_conditions -ambient_temp 25 -process typical

report_power -hierarchical_depth 4 -file $report_dir/power.rpt
foreach rate {6.25 25} {
    set_switching_activity -default_toggle_rate $rate
    report_power -hierarchical_depth 4 -file $report_dir/power_toggle$rate.rpt
}
set_switching_activity -default_toggle_rate 12.5
puts "POWER REPORTS: $report_dir"
