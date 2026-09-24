# build.tcl — full Vivado build for the minimal RISC-V / SNN design (PYNQ-Z2).
# Usage: ./vivado2024.sh -mode batch -source build.tcl   (run from vivado/)
#
# Produces ./spike_top.bit (copied out of the run directory). The PYNQ-Z2
# board files are vendored in ./board_files (TUL pynq-z2, from
# Xilinx/XilinxBoardStore). Vivado 2024.1 has the Zynq-7000 parts.

set part xc7z020clg400-1
set board tul.com.tw:pynq-z2:part0:1.0
set projdir [file normalize ./rvproj]
if {$argc >= 1} { set projdir [file normalize [lindex $argv 0]] }
set output_bit [file normalize ./spike_top.bit]
if {$argc >= 2} { set output_bit [file normalize [lindex $argv 1]] }

# ----------------------------------------------------------------------
# Project
# ----------------------------------------------------------------------
set_param board.repoPaths [list [file normalize ./board_files]]

if {[file exists $projdir]} { error "Build directory already exists: $projdir. Choose a fresh path with -tclargs." }
create_project rv $projdir -part $part -force
set_property board_part $board [current_project]

# Add every RTL source (so a new rtl/*.v is picked up automatically).
add_files -norecurse [glob ../rtl/*.v]
add_files -fileset constrs_1 -norecurse ./spike_top.xdc

# ----------------------------------------------------------------------
# Block design: PS7 (GP0 master, FCLK0) -> interconnect -> protocol
# converter -> external AXI4-Lite master port M00_AXI
# ----------------------------------------------------------------------
create_bd_design ps_bd

set ps7 [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7]
foreach interface {DDR FIXED_IO} {
    make_bd_intf_pins_external [get_bd_intf_pins ps7/$interface]
    set_property name $interface [get_bd_intf_ports ${interface}_0]
}

# Board preset (DDR, MIO, clocks for the PYNQ-Z2): the vendored board file
# preset.xml is a plain list of CONFIG parameters — apply them directly.
set preset_xml [file normalize ./board_files/pynq-z2/A.0/preset.xml]
set fp [open $preset_xml r]
set xml [read $fp]
close $fp
set params {}
foreach {- name value} [regexp -all -inline {name="(CONFIG\.[^"]+)"\s+value="([^"]*)"} $xml] {
    if {![string match CONFIG.PCW_* $name]} continue
    if {[string match PCW_*_AXI_*FREQMHZ $name]} continue
    lappend params $name $value
}
puts "board preset: applying [llength $params]/2 PS7 parameters"
set_property -dict $params [get_bd_cells ps7]

# Only GP0 and FCLK0 matter at runtime.
set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_M_AXI_GP1 {0} \
    CONFIG.PCW_EN_CLK0_PORT {1} \
    CONFIG.PCW_EN_RST0_PORT {0} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100.000000} \
] $ps7

set ic [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 ic0]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] $ic

set pc [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_protocol_converter:2.1 pc0]

connect_bd_intf_net [get_bd_intf_pins ps7/M_AXI_GP0] [get_bd_intf_pins ic0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins ic0/M00_AXI]   [get_bd_intf_pins pc0/S_AXI]

# Clocks: everything on FCLK0
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins ic0/ACLK]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins ic0/S00_ACLK]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins ic0/M00_ACLK]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins ps7/M_AXI_GP0_ACLK]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins pc0/aclk]

# Reset: one external active-low port, driven by the POR in spike_top
set rst_port [create_bd_port -dir I aresetn]
connect_bd_net $rst_port [get_bd_pins ic0/ARESETN] [get_bd_pins pc0/aresetn]
foreach p {ic0/S00_ARESETN ic0/M00_ARESETN} {
    if {[llength [get_bd_pins -quiet $p]]} {
        connect_bd_net $rst_port [get_bd_pins $p]
    }
}

# External AXI4-Lite master port for the PL (name it deterministically)
set if_ports_before [get_bd_intf_ports -quiet]
make_bd_intf_pins_external [get_bd_intf_pins pc0/M_AXI]
set new_if_port ""
foreach p [get_bd_intf_ports] {
    if {[lsearch -exact $if_ports_before $p] < 0} { set new_if_port $p }
}
set_property name M00_AXI $new_if_port

# FCLK0 out for spike_top
set fclk_port [create_bd_port -dir O FCLK_CLK0]
connect_bd_net $fclk_port [get_bd_pins ps7/FCLK_CLK0]

# the address segment lives on the EXTERNAL interface port we just made
set ext_seg [get_bd_addr_segs -quiet M00_AXI/Reg]
if {[llength $ext_seg] == 0} { set ext_seg [get_bd_addr_segs -quiet *Reg*] }
assign_bd_address -offset 0x40000000 -range 0x800000 $ext_seg

validate_bd_design
save_bd_design

# Wrapper for the BD (spike_top instantiates it)
set_property top spike_top [current_fileset]
make_wrapper -files [get_files ps_bd.bd] -top -import -force
update_compile_order -fileset sources_1

# ----------------------------------------------------------------------
# Synthesis + implementation + bitstream
# ----------------------------------------------------------------------
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1

set bit $projdir/rv.runs/impl_1/spike_top.bit
if {[file exists $bit]} {
    file copy -force $bit $output_bit
    open_run impl_1
    report_timing_summary -file $projdir/timing_summary.rpt
    report_utilization -file $projdir/utilization.rpt
    report_drc -file $projdir/drc.rpt
    puts "BITSTREAM: $output_bit"
} else {
    error "bitstream not produced - check runs"
}
