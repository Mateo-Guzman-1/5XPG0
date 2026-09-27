# Run the streaming (ABI v3) vectors on the board over JTAG (after jtag/bringup.tcl
# with a keyword_stream*.bin firmware).
# Usage: python jtag/make_stream_vectors.py --model <int model.npz>
#        xsdb jtag/stream_board_test.tcl [vectors.tcl] [results.csv]
# Per stream: reset (command 5), then one command 4 per hop. Checks best/last
# scores, spikes, detection bits and detection frame against the integer oracle,
# the frame counter, a malformed request, and the LED0 pulse length (polling
# syscon over JTAG).
set root [file normalize [file join [file dirname [info script]] ..]]
set vectors [expr {[llength $argv] > 0 ? [lindex $argv 0] : "$root/build/stream_board_vectors.tcl"}]
set csv [expr {[llength $argv] > 1 ? [lindex $argv 1] : "$root/build/stream_board_jtag.csv"}]
connect -url tcp:127.0.0.1:3121
# APU debug port (physical addresses) when PYNQ Linux runs with the MMU on,
# otherwise the halted core as after the ps7_init path of jtag/bringup.tcl.
targets -set -filter {name =~ "ARM*#0"}
configparams force-mem-access 1
if {[catch {mrd -force 0x40040014}]} {targets -set -filter {name =~ "APU"}}
proc rd {addr} {return [expr {wide([lindex [mrd -force -value $addr] 0])}]}
proc signed {n} { if {$n >= 0x80000000} {return [expr {$n-0x100000000}]}; return $n }
proc mb {i} {return [expr {0x40010400 + 4*$i}]}
set seq [rd [mb 0]]
proc command {opcode length} {
    global seq
    mwr -force [mb 2] [list $opcode $length]
    set seq [expr {($seq+1)&0xffffffff}]
    mwr -force [mb 0] $seq
    set deadline [expr {[clock milliseconds]+3000}]
    while {[rd [mb 1]] != $seq} {
        if {[rd 0x40040004]&2} {error "PicoRV32 trap"}
        if {[clock milliseconds]>$deadline} {error "Firmware timeout"}
        after 2
    }
    return [rd [mb 6]]
}
source $vectors
if {[rd [mb 12]] != 0x4b575333} {error "Streaming firmware (KWS3) not running"}
if {[rd [mb 13]] != $threshold} {error "Firmware threshold [rd [mb 13]] != model $threshold"}
set fb [rd [mb 14]]
if {[command 4 [expr {$fb-1}]] != 1} {error "Partial frame accepted"}
if {[command 4 [expr {$fb*([rd [mb 15]]+1)}]] != 1} {error "Too many frames accepted"}
if {[command 9 0] != 1} {error "Unknown opcode accepted"}
set out [open $csv w]
puts $out "record,hop,frames,best,last,spikes,events,cycles,detected"
set record 0
set total 0
set led_checked 0
foreach stream $streams {
    if {[command 5 0] != 0 || [rd [mb 18]] != 0} {error "Reset failed"}
    set frames 0
    set hop 0
    foreach h $stream {
        lassign $h k best last spikes detected at words
        mwr -force 0x40010800 $words
        if {[command 4 [expr {$k*$fb}]] != 0} {error "Stream request failed"}
        incr frames $k
        set got [list [signed [rd [mb 7]]] [signed [rd [mb 8]]] [rd [mb 10]] [rd [mb 11]] [rd [mb 17]] [rd [mb 18]]]
        set want [list $best $last $spikes $detected $at $frames]
        if {$got ne $want} {error "Mismatch record $record hop $hop: got $got expected $want"}
        set cycles [rd [mb 9]]
        puts $out "$record,$hop,$k,$best,$last,$spikes,[rd [mb 16]],$cycles,$detected"
        flush $out
        if {($detected & 1) && !$led_checked} {
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
        incr hop
        incr total
    }
    puts "PASS stream=$record hops=$hop frames=$frames"
    incr record
}
close $out
if {!$led_checked} {error "No LED test"}
puts "PASS: $record streams, $total hops match the integer oracle (malformed requests rejected, reset, LED)."
puts "FINAL STATUS: [mrd -force 0x40040000 8]"
exit
