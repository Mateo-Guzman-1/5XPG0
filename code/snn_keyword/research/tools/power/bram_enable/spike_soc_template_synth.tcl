read_verilog {spike_soc.v kdot_pcpi.v picorv32.v poisson.v}
synth_design -top spike_soc -part xc7z020clg400-1 -mode out_of_context -generic POWER_OPT=1 -generic READ_WAIT=1
set fh [open en_report.txt w]
proc en_report {fh tag} {
    set mem [lsort [get_cells -hier -filter {REF_NAME =~ RAMB* && NAME =~ *mem_reg*}]]
    puts $fh "== $tag: [llength $mem] blocks"
    set seen {}
    foreach r $mem {
        set ext [get_property RAM_EXTENSION_A $r]
        if {[lsearch $seen $ext] >= 0} continue
        lappend seen $ext
        foreach p {ENARDEN ENBWREN} {
            set drv [get_cells -quiet -of [get_pins -quiet -leaf -of [get_nets -of [get_pins $r/$p]] -filter {DIRECTION == OUT}]]
            set ins {}
            foreach ip [get_pins -quiet -of $drv -filter {DIRECTION == IN}] { lappend ins [get_nets -quiet -of $ip] }
            puts $fh "  $r $ext $p <- [get_property REF_NAME $drv] INIT=[get_property -quiet INIT $drv] inputs: $ins"
        }
    }
}
en_report $fh synth
opt_design
en_report $fh opt
puts $fh "utilization: [llength [get_cells -hier -filter {REF_NAME =~ LUT*}]] LUTs"
close $fh
