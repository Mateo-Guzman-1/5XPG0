# One-command PYNQ demo: upload, load bitstream, start board_server.py, run
# pc_keyword_demo.py, and stop the server again when the demo exits.
# The first run installs an SSH key on the board (asks for the password once).
[CmdletBinding()]
param(
    [string]$Board = "192.168.2.99",
    [string]$User = "xilinx",
    [string]$Password = "xilinx",   # board sudo password (PYNQ default)
    [string]$Python = "python",
    [switch]$Kdot,                  # use keyword_kdot.bit/.bin instead of keyword
    [switch]$Single,                # passed on to pc_keyword_demo.py
    [string]$Wav,                   # replay one 16 kHz mono WAV instead of the microphone
    [switch]$KeepServer             # leave board_server.py running afterwards
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$Target = "$User@$Board"
$Dir = "/home/$User/snn_keyword"
$Image = if ($Kdot) { "keyword_kdot" } else { "keyword" }
$Ssh = @("-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10")

function Invoke-Board([string]$Command) {
    & ssh @Ssh -o BatchMode=yes $Target $Command
    if ($LASTEXITCODE -ne 0) { throw "Board command failed: $Command" }
}

# 1. Password-free SSH (one-time setup).
& ssh @Ssh -o BatchMode=yes -o LogLevel=QUIET $Target true
if ($LASTEXITCODE -ne 0) {
    $Key = Join-Path $env:USERPROFILE ".ssh\id_ed25519"
    if (-not (Test-Path -LiteralPath "$Key.pub")) {
        New-Item -ItemType Directory -Force (Split-Path $Key) | Out-Null
        # ProcessStartInfo passes the empty passphrase reliably on PowerShell 5.1 and 7.
        $psi = New-Object System.Diagnostics.ProcessStartInfo "ssh-keygen", "-q -t ed25519 -N `"`" -f `"$Key`""
        $psi.UseShellExecute = $false
        $p = [System.Diagnostics.Process]::Start($psi); $p.WaitForExit()
        if ($p.ExitCode -ne 0) { throw "ssh-keygen failed" }
    }
    $Pub = (Get-Content -LiteralPath "$Key.pub" -Raw).Trim()
    Write-Host "Installing SSH key on $Board; enter the board password ($Password) once."
    & ssh @Ssh $Target "mkdir -p ~/.ssh && chmod 700 ~/.ssh && echo '$Pub' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
    if ($LASTEXITCODE -ne 0) { throw "Could not install the SSH key on $Board" }
}

# 2. Upload the board-side files (a few MB, keeps the board in sync with this folder).
Write-Host "Uploading to ${Target}:$Dir"
Invoke-Board "mkdir -p $Dir"
& scp @Ssh -o BatchMode=yes -q -r deploy board_server.py protocol.py features.py ../../board_run.sh "${Target}:$Dir/"
if ($LASTEXITCODE -ne 0) { throw "Upload failed" }

# 3. Load the bitstream and start the server (sudo -i loads the PYNQ environment).
Write-Host "Loading $Image.bit and starting board_server.py"
Invoke-Board "sed -i 's/\r$//' $Dir/board_run.sh && echo '$Password' | sudo -S -p '' -i bash $Dir/board_run.sh $Image"

# 4. Run the demo on this PC; stop the server when it ends (also on Ctrl+C).
try {
    $DemoArgs = @("pc_keyword_demo.py", $Board)
    if ($Single) { $DemoArgs += "--single" }
    if ($Wav) { $DemoArgs += @("--wav", $Wav) }
    & $Python @DemoArgs
} finally {
    if (-not $KeepServer) {
        Write-Host "Stopping board_server.py"
        & ssh @Ssh -o BatchMode=yes $Target "echo '$Password' | sudo -S -p '' pkill -f board_server.py"
    }
}
