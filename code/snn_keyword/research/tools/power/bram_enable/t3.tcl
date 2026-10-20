set fh [open report3.txt w]
read_verilog ram_be2.v
synth_design -top ram_be2 -part xc7z020clg400-1 -mode out_of_context -generic AW=16
set mem [lsort [get_cells -hier -filter {REF_NAME =~ RAMB*}]]
puts $fh "== ram_be2: [llength $mem] blocks FFs [llength [get_cells -hier -filter {REF_NAME =~ FD*}]] LUTs [llength [get_cells -hier -filter {REF_NAME =~ LUT*}]]"
foreach r [lrange $mem 0 0] {
    puts $fh "  $r [get_property RAM_EXTENSION_A $r] RA=[get_property READ_WIDTH_A $r] WMODE=[get_property WRITE_MODE_A $r]"
    foreach p {ENARDEN ENBWREN} {
        set net [get_nets -quiet -of [get_pins $r/$p]]
        set drv [get_cells -quiet -of [get_pins -quiet -leaf -of $net -filter {DIRECTION == OUT}]]
        set ins {}
        foreach ip [get_pins -quiet -of $drv -filter {DIRECTION == IN}] { lappend ins [get_nets -quiet -of $ip] }
        puts $fh "  $p net $net <- [get_property -quiet REF_NAME $drv] [get_property -quiet INIT $drv] $ins"
    }
}
close $fh
