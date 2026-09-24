# Run the 40 verification vectors on the board over JTAG (after jtag/bringup.tcl).
# Usage: python jtag/make_board_vectors.py --model <model.npz>
#        xsdb jtag/board_test.tcl [vectors.tcl] [results.csv]
# Checks scores/spikes against the integer oracle, the decision against the
# model threshold, and the LED0 pulse length (by polling syscon over JTAG).
set root [file normalize [file join [file dirname [info script]] ..]]
set vectors [expr {[llength $argv] > 0 ? [lindex $argv 0] : "$root/build/board_vectors.tcl"}]
set csv [expr {[llength $argv] > 1 ? [lindex $argv 1] : "$root/build/board_jtag.csv"}]
connect -url tcp:127.0.0.1:3121
targets -set -filter {name =~ "ARM*#0"}
configparams force-mem-access 1
proc rd {addr} {
    set value [lindex [mrd -force -value $addr] 0]
    return [expr {wide($value)}]
}
proc signed {n} { if {$n >= 0x80000000} {return [expr {$n-0x100000000}]}; return $n }
source $vectors
set input_bytes [rd 0x40010438]
if {$input_bytes == 0} {set input_bytes 768}
set out [open $csv w]
puts $out "index,score0,score1,spikes,cycles,detected"
set seq [rd 0x40010400]
set index 0
set led_checked 0
foreach frame $frames {
    lassign $frame expected0 expected1 expected_spikes words
    mwr -force 0x40010800 $words
    mwr -force 0x40010408 [list 1 $input_bytes]
    set seq [expr {($seq+1)&0xffffffff}]
    mwr -force 0x40010400 $seq
    set deadline [expr {[clock milliseconds]+3000}]
    while {[rd 0x40010404] != $seq} {
        if {[rd 0x40040004]&2} {error "PicoRV32 trap"}
        if {[clock milliseconds]>$deadline} {error "Firmware timeout"}
        after 5
    }
    if {[rd 0x40010418] != 0} {error "Firmware error"}
    set s0 [signed [rd 0x4001041c]]
    set s1 [signed [rd 0x40010420]]
    set cycles [rd 0x40010424]
    set spikes [rd 0x40010428]
    set detected [rd 0x4001042c]
    if {$s0!=$expected0 || $s1!=$expected1 || $spikes!=$expected_spikes} {
        error "Mismatch $index: $s0,$s1,$spikes expected $expected0,$expected1,$expected_spikes"
    }
    if {$detected != ($s1-$s0>=$threshold)} {error "Decision mismatch"}
    puts $out "$index,$s0,$s1,$spikes,$cycles,$detected"
    flush $out
    puts "PASS vector=$index scores=$s0,$s1 cycles=$cycles detected=$detected"
    if {$detected && !$led_checked} {
        if {[rd 0x40040018] != 1} {error "LED did not turn on"}
        set start [clock milliseconds]
        while {[rd 0x40040018] != 0} {
            if {[clock milliseconds]-$start > 1500} {error "LED stayed on too long"}
            after 5
        }
        set duration [expr {[clock milliseconds]-$start}]
        if {$duration < 850 || $duration > 1100} {error "Unexpected LED duration $duration ms"}
        puts "LED physical register observation: $duration ms from post-inference read to off (JTAG polling)"
        set led_checked 1
    }
    incr index
}
close $out
if {!$led_checked} {error "No LED test"}
puts "PASS: all $index physical-board vectors match the independent oracle."
puts "FINAL STATUS: [mrd -force 0x40040000 8]"
exit
