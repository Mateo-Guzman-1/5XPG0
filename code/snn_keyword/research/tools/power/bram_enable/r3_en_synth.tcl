# BRAM enables after synthesis (before opt_design's BRAM power optimization) and after implementation.
set out [lindex $argv 0]
set fh [open $out w]
foreach {tag dcp} [lrange $argv 1 end] {
    open_checkpoint $dcp
    set mem [lsort [get_cells -hier -filter {REF_NAME =~ RAMB* && NAME =~ u_soc/mem_reg*}]]
    puts $fh "== $tag ($dcp): [llength $mem] blocks"
    foreach r [lrange $mem 0 1] {
        puts $fh "$r EXT_A=[get_property RAM_EXTENSION_A $r] WMODE_A=[get_property WRITE_MODE_A $r] WMODE_B=[get_property WRITE_MODE_B $r] DOA_REG=[get_property DOA_REG $r]"
        foreach p {ENARDEN ENBWREN REGCEAREGCE REGCEB} {
            set net [get_nets -quiet -of [get_pins $r/$p]]
            set drv [get_cells -quiet -of [get_pins -quiet -leaf -of $net -filter {DIRECTION == OUT}]]
            set ins {}
            foreach ip [get_pins -quiet -of $drv -filter {DIRECTION == IN}] {
                lappend ins "[get_property REF_PIN_NAME $ip]=[get_nets -quiet -of $ip]"
            }
            puts $fh "  $p <- [get_property REF_NAME $drv] $drv INIT=[get_property -quiet INIT $drv] inputs: $ins"
        }
    }
    puts $fh "pb_rdata / pa_rdata registers: [llength [get_cells -quiet -hier -filter {NAME =~ *pb_rdata* || NAME =~ *pa_rdata_i*}]]"
    foreach c [lrange [get_cells -quiet -hier -filter {NAME =~ *pb_rdata* || NAME =~ *pa_rdata_i*}] 0 3] { puts $fh "  $c [get_property REF_NAME $c]" }
    close_design
}
close $fh
