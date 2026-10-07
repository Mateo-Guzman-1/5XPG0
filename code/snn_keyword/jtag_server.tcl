# Volatile debug transport for an already programmed/initialized PYNQ-Z2.
connect -url tcp:127.0.0.1:3121
# The hw_server can invalidate the CPU target context (e.g. when Vivado's
# Hardware Manager refreshes); re-select it and retry the frame once.
# With PYNQ Linux running (MMU on), use the APU debug port (physical addresses).
proc select_cpu {} {
    targets -set -filter {name =~ "ARM*#0"}
    configparams force-mem-access 1
    if {[catch {mrd -force 0x40040014}]} {targets -set -filter {name =~ "APU"}}
}
select_cpu
proc rd {address} {return [expr {wide([lindex [mrd -force -value $address] 0])}]}
proc signed {n} {if {$n>=0x80000000} {return [expr {$n-0x100000000}]}; return $n}
# Firmware magic: "KWS1" = ABI v2 (window model), "KWS3" = ABI v3 (streaming model).
set magic [rd 0x40010430]
if {([rd 0x4004001c]&0xffff0000)!=0x20000 || ($magic!=0x4b575331 && $magic!=0x4b575333)} {error "Load the keyword bitstream and firmware first"}
set abi [expr {$magic==0x4b575333 ? 3 : 2}]
# v2: MB[14] = input size (older images: 0 = 768 bytes). v3: MB[14] = frame bytes, MB[15] = max frames.
set input_bytes [rd 0x40010438]
if {$input_bytes == 0} {set input_bytes 768}
set max_frames [expr {$abi==3 ? [rd 0x4001043c] : 0}]
proc run_frame {opcode line {length -1}} {
    if {$length < 0} {set length $::input_bytes}
    if {$length > 0} {
        binary scan [binary format H* $line] i* raw
        set words {}
        foreach word $raw {lappend words [expr {$word&0xffffffff}]}
        mwr -force 0x40010800 $words
    }
    mwr -force 0x40010408 [list $opcode $length]
    set seq [expr {([rd 0x40010400]+1)&0xffffffff}]
    mwr -force 0x40010400 $seq
    set deadline [expr {[clock milliseconds]+3000}]
    while {[rd 0x40010404]!=$seq} {
        if {[rd 0x40040004]&2} {error "CPU trap"}
        if {[clock milliseconds]>$deadline} {error "Firmware timeout"}
        after 2
    }
    set r [list [rd 0x40010418] [signed [rd 0x4001041c]] [signed [rd 0x40010420]] [rd 0x40010424] [rd 0x40010428] [rd 0x4001042c]]
    # v3 adds synaptic events (MB[16]) and the frame of the detection (MB[17], -1 = none).
    if {$::abi == 3} {lappend r [rd 0x40010440] [signed [rd 0x40010444]]}
    return $r
}
proc process_frame {channel} {
    if {[eof $channel]} {close $channel;return}
    if {[gets $channel line]<0} {return}
    # v2 line: "<1|3> <hex window>"; 1 = single window, 3 = stream window ("2 of 3").
    # v3 lines: "I" (info), "4 <hex frames>" (whole frames), "5" (reset).
    # Tcl regexps cap repetition counts at 255, so lengths are checked separately.
    if {$line eq "I"} {
        puts $channel "INFO $::abi $::input_bytes [signed [rd 0x40010434]] $::max_frames";flush $channel;return
    }
    set length -1
    if {$::abi == 2} {
        if {![regexp {^([13]) ([0-9a-f]+)$} $line -> opcode line] || [string length $line]!=2*$::input_bytes} {
            puts $channel "ERROR invalid_feature_frame";flush $channel;return
        }
    } elseif {$line eq "5"} {
        set opcode 5; set line ""; set length 0
    } else {
        set frame_hex [expr {2*$::input_bytes}]
        if {![regexp {^4 ([0-9a-f]+)$} $line -> line] || [string length $line] % $frame_hex
            || [string length $line] == 0 || [string length $line] > $frame_hex*$::max_frames} {
            puts $channel "ERROR invalid_feature_frame";flush $channel;return
        }
        set opcode 4; set length [expr {[string length $line]/2}]
    }
    set failed [catch {run_frame $opcode $line $length} result]
    # The cable can also vanish briefly ("no targets found"); wait up to 5 s for it.
    set retry_until [expr {[clock milliseconds]+5000}]
    while {$failed && ([string match "*Invalid context*" $result] || [string match "*no targets found*" $result])
           && [clock milliseconds] < $retry_until} {
        puts stderr "JTAG target lost ($result); re-selecting CPU target"
        after 250
        set failed [catch {select_cpu; run_frame $opcode $line $length} result]
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
