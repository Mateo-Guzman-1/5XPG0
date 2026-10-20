# What drives the BRAM enables of a routed R3 checkpoint: inputs and INIT of the enable LUTs of a LOWER and an
# UPPER block of the same bit (Vivado power optimization "cooolgate").
set dcp [lindex $argv 0]
set out [lindex $argv 1]
open_checkpoint $dcp
set mem [lsort [get_cells -hier -filter {REF_NAME =~ RAMB* && NAME =~ u_soc/mem_reg*}]]
set fh [open $out w]
set n_upper 0
foreach r $mem { if {[get_property RAM_EXTENSION_A $r] eq "UPPER"} { incr n_upper } }
puts $fh "blocks [llength $mem], UPPER $n_upper"
set lower [lindex $mem 0]
set upper ""
foreach r $mem { if {[get_property RAM_EXTENSION_A $r] eq "UPPER"} { set upper $r; break } }
foreach r [list $lower $upper] {
    puts $fh "$r [get_property RAM_EXTENSION_A $r] RA=[get_property READ_WIDTH_A $r]"
    foreach p {ENARDEN ENBWREN} {
        set drv [get_cells -of [get_pins -leaf -of [get_nets -of [get_pins $r/$p]] -filter {DIRECTION == OUT}]]
        puts $fh "  $p <- [get_property REF_NAME $drv] $drv INIT=[get_property INIT $drv]"
        foreach ip [get_pins -of $drv -filter {DIRECTION == IN}] {
            set net [get_nets -of $ip]
            set src [get_pins -quiet -leaf -of $net -filter {DIRECTION == OUT}]
            puts $fh "    [get_property REF_PIN_NAME $ip] <- $net (from $src)"
        }
    }
}
close $fh
close_design
