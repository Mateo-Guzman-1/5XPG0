# Volatile debug transport for an already programmed/initialized PYNQ-Z2.
connect -url tcp:127.0.0.1:3121
# The hw_server can invalidate the CPU target context (e.g. when Vivado's
# Hardware Manager refreshes); re-select it and retry the frame once.
proc select_cpu {} {
    targets -set -filter {name =~ "ARM*#0"}
    configparams force-mem-access 1
}
select_cpu
proc rd {address} {return [expr {wide([lindex [mrd -force -value $address] 0])}]}
proc signed {n} {if {$n>=0x80000000} {return [expr {$n-0x100000000}]}; return $n}
if {([rd 0x4004001c]&0xffff0000)!=0x20000 ||[rd 0x40010430]!=0x4b575331} {error "Load the keyword bitstream and firmware first"}
# Firmware publishes its input size in MB[14]; older images (0) use 768 bytes.
set input_bytes [rd 0x40010438]
if {$input_bytes == 0} {set input_bytes 768}
proc run_frame {opcode line} {
    binary scan [binary format H* $line] i* raw
    set words {}
    foreach word $raw {lappend words [expr {$word&0xffffffff}]}
    mwr -force 0x40010800 $words
    mwr -force 0x40010408 [list $opcode $::input_bytes]
    set seq [expr {([rd 0x40010400]+1)&0xffffffff}]
    mwr -force 0x40010400 $seq
    set deadline [expr {[clock milliseconds]+3000}]
    while {[rd 0x40010404]!=$seq} {
        if {[rd 0x40040004]&2} {error "CPU trap"}
        if {[clock milliseconds]>$deadline} {error "Firmware timeout"}
        after 2
    }
    return [list [rd 0x40010418] [signed [rd 0x4001041c]] [signed [rd 0x40010420]] [rd 0x40010424] [rd 0x40010428] [rd 0x4001042c]]
}
proc process_frame {channel} {
    if {[eof $channel]} {close $channel;return}
    if {[gets $channel line]<0} {return}
    # Line: "<opcode> <hex features>"; 1 = single window, 3 = stream window ("2 of 3").
    # Tcl regexps cap repetition counts at 255, so check the length separately.
    if {![regexp {^([13]) ([0-9a-f]+)$} $line -> opcode line] || [string length $line]!=2*$::input_bytes} {
        puts $channel "ERROR invalid_feature_frame";flush $channel;return
    }
    set failed [catch {run_frame $opcode $line} result]
    # The cable can also vanish briefly ("no targets found"); wait up to 5 s for it.
    set retry_until [expr {[clock milliseconds]+5000}]
    while {$failed && ([string match "*Invalid context*" $result] || [string match "*no targets found*" $result])
           && [clock milliseconds] < $retry_until} {
        puts stderr "JTAG target lost ($result); re-selecting CPU target"
        after 250
        set failed [catch {select_cpu; run_frame $opcode $line} result]
    }
    if {$failed} {
        catch {mwr -force 0x40040000 1}
        puts $channel "ERROR $result"
        flush $channel
        close $channel
        return
    }
    puts $channel "RESULT $result"
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
