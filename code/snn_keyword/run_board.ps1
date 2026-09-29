# Start the keyword demo on a PYNQ-Z2 running PYNQ Linux (standard Ethernet path).
# Copies the board-side files over SSH, loads the bitstream and starts
# board_server.py in the background (start_board.sh). Needs passwordless SSH.
# Usage:  .\run_board.ps1 [-Board pynq] [-Variant kdot|base|stop]
# Then:   .venv/Scripts/python.exe code/snn_keyword/pc_keyword_demo.py <board-ip>
# Nothing persists on the FPGA: run this again after every board reboot.
param(
    [string]$Board = 'pynq',   # SSH host: an ~/.ssh/config alias or user@ip
    [ValidateSet('kdot', 'base', 'stop')][string]$Variant = 'kdot'
)
$ErrorActionPreference = 'Stop'

function Invoke-Checked {
    & $args[0] @($args | Select-Object -Skip 1)
    if ($LASTEXITCODE) { throw "$($args[0]) failed with exit code $LASTEXITCODE" }
}

if ($Variant -ne 'stop') {
    Invoke-Checked ssh $Board 'mkdir -p snn_keyword/deploy'
    Invoke-Checked scp -q "$PSScriptRoot/board_server.py" "$PSScriptRoot/protocol.py" `
        "$PSScriptRoot/features.py" "$PSScriptRoot/start_board.sh" "${Board}:snn_keyword/"
    Invoke-Checked scp -q "$PSScriptRoot/deploy/keyword.bit" "$PSScriptRoot/deploy/keyword.bin" `
        "$PSScriptRoot/deploy/keyword_kdot.bit" "$PSScriptRoot/deploy/keyword_kdot.bin" "${Board}:snn_keyword/deploy/"
}
Invoke-Checked ssh $Board "bash snn_keyword/start_board.sh $Variant"
