# Volatile debug transport for an already programmed/initialized PYNQ-Z2.
connect -url tcp:127.0.0.1:3121
targets -set -filter {name =~ "ARM*#0"}
configparams force-mem-access 1
proc rd {address} {return [expr {wide([lindex [mrd -force -value $address] 0])}]}
proc signed {n} {if {$n>=0x80000000} {return [expr {$n-0x100000000}]}; return $n}
if {([rd 0x4004001c]&0xffff0000)!=0x20000 ||[rd 0x40010430]!=0x4b575331} {error "Load the keyword bitstream and firmware first"}
proc process_frame {channel} {
    if {[eof $channel]} {close $channel;return}
    if {[gets $channel line]<0} {return}
    # Tcl regexps cap repetition counts at 255, so check the length separately.
    if {[string length $line]!=1536 || ![regexp {^[0-9a-f]+$} $line]} {puts $channel "ERROR invalid_feature_frame";flush $channel;return}
    if {[catch {
        binary scan [binary format H* $line] i* raw
        set words {}
        foreach word $raw {lappend words [expr {$word&0xffffffff}]}
        mwr -force 0x40010800 $words
        mwr -force 0x40010408 {1 768}
        set seq [expr {([rd 0x40010400]+1)&0xffffffff}]
        mwr -force 0x40010400 $seq
        set deadline [expr {[clock milliseconds]+3000}]
        while {[rd 0x40010404]!=$seq} {
            if {[rd 0x40040004]&2} {error "CPU trap"}
            if {[clock milliseconds]>$deadline} {error "Firmware timeout"}
            after 2
        }
        set result [list [rd 0x40010418] [signed [rd 0x4001041c]] [signed [rd 0x40010420]] [rd 0x40010424] [rd 0x40010428] [rd 0x4001042c]]
        puts $channel "RESULT $result"
    } message]} {
        catch {mwr -force 0x40040000 1}
        puts $channel "ERROR $message"
        flush $channel
        close $channel
        return
    }
    flush $channel
}
proc accept_client {channel address port} {
    fconfigure $channel -blocking 0 -buffering line -translation lf
    fileevent $channel readable [list process_frame $channel]
}
set listener [socket -server accept_client -myaddr 127.0.0.1 5557]
puts READY
flush stdout
vwait forever
exit
