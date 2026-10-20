set fh [open report.txt w]
foreach aw {16 14} {
    read_verilog ram.v
    synth_design -top ram -part xc7z020clg400-1 -mode out_of_context -generic AW=$aw
    set mem [lsort [get_cells -hier -filter {REF_NAME =~ RAMB*}]]
    puts $fh "== AW=$aw: [llength $mem] blocks, [get_property RAM_EXTENSION_A [lindex $mem 0]] RA=[get_property READ_WIDTH_A [lindex $mem 0]]"
    foreach r [lrange $mem 0 1] {
        foreach p {ENARDEN ENBWREN} {
            set drv [get_cells -quiet -of [get_pins -quiet -leaf -of [get_nets -of [get_pins $r/$p]] -filter {DIRECTION == OUT}]]
            set ins {}
            foreach ip [get_pins -quiet -of $drv -filter {DIRECTION == IN}] { lappend ins [get_nets -quiet -of $ip] }
            puts $fh "  $r [get_property RAM_EXTENSION_A $r] $p <- [get_property REF_NAME $drv] [get_property -quiet INIT $drv] $ins"
        }
    }
    puts $fh "  FFs: [llength [get_cells -hier -filter {REF_NAME =~ FD*}]] LUTs: [llength [get_cells -hier -filter {REF_NAME =~ LUT*}]]"
    close_design
}
close $fh
