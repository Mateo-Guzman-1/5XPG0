set fh [open report2.txt w]
foreach {top file} {ram_be ram_be.v ram_be2 ram_be2.v} {
    read_verilog $file
    synth_design -top $top -part xc7z020clg400-1 -mode out_of_context -generic AW=16
    set mem [lsort [get_cells -hier -filter {REF_NAME =~ RAMB*}]]
    puts $fh "== $top: [llength $mem] blocks [get_property RAM_EXTENSION_A [lindex $mem 0]] RA=[get_property READ_WIDTH_A [lindex $mem 0]] FFs [llength [get_cells -hier -filter {REF_NAME =~ FD*}]] LUTs [llength [get_cells -hier -filter {REF_NAME =~ LUT*}]]"
    foreach r [lrange $mem 0 0] {
        foreach p {ENARDEN ENBWREN ADDRARDADDR[3] DOADO[0]} {
            set net [get_nets -quiet -of [get_pins $r/$p]]
            set drv [get_cells -quiet -of [get_pins -quiet -leaf -of $net -filter {DIRECTION == OUT}]]
            set ins {}
            foreach ip [get_pins -quiet -of $drv -filter {DIRECTION == IN}] { lappend ins [get_nets -quiet -of $ip] }
            puts $fh "  $p net $net <- [get_property REF_NAME $drv] [get_property -quiet INIT $drv] $ins"
        }
    }
    close_design
}
close $fh
