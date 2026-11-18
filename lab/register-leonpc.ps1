# Register this repo with a Cursor worker in Windows PowerShell 5.1.
# The lab tunnel and the cbu_lab key are on Windows, so this is the shell
# that can submit jobs. Leave the robot worker running in its own shell.
#
#   powershell -ExecutionPolicy Bypass -File "$env:USERPROFILE\ridcheck\lab\register-leonpc.ps1"
#   powershell -ExecutionPolicy Bypass -File "$env:USERPROFILE\ridcheck\lab\register-leonpc.ps1" -Start

param(
    [switch]$Start
)

$ErrorActionPreference = 'Stop'
$Root = Join-Path $env:USERPROFILE 'ridcheck'
$Name = 'LeonPC-ridcheck'
$Key = Join-Path $env:USERPROFILE '.ssh\cbu_lab'

function Find-Agent {
    $cmd = Get-Command agent -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    $dirs = @(
        (Join-Path $env:USERPROFILE '.local\bin'),
        (Join-Path $env:LOCALAPPDATA 'cursor-agent')
    )
    foreach ($dir in $dirs) {
        foreach ($file in @('agent.exe', 'agent.cmd', 'agent.ps1')) {
            $path = Join-Path $dir $file
            if (Test-Path $path) {
                return $path
            }
        }
    }
    return $null
}

$stray = Join-Path $env:USERPROFILE '~\ridcheck'
if (-not (Test-Path (Join-Path $Root '.git'))) {
    if (Test-Path -LiteralPath (Join-Path $stray '.git')) {
        Write-Host "Moving $stray to $Root"
        Move-Item -LiteralPath $stray -Destination $Root
    } else {
        git clone https://github.com/leonjorge010/ridcheck.git $Root
    }
} else {
    git -C $Root pull --ff-only
}

if (-not (Test-Path $Key)) {
    Write-Host "Missing $Key. That is the lab key the portal already has on file."
    exit 1
}

Write-Host 'Checking the lab login node'
$client = New-Object System.Net.Sockets.TcpClient
try {
    $wait = $client.BeginConnect('10.77.0.1', 2222, $null, $null)
    $opened = $wait.AsyncWaitHandle.WaitOne(5000, $false) -and $client.Connected
} finally {
    $client.Close()
}
if (-not $opened) {
    Write-Host 'This PC cannot reach 10.77.0.1:2222. Activate the cbu_lab tunnel in WireGuard, then run this again.'
    exit 1
}
Write-Host 'Lab login node is reachable'

$agent = Find-Agent
if (-not $agent) {
    Write-Host 'Installing the Cursor CLI'
    Invoke-Expression (Invoke-RestMethod 'https://cursor.com/install?win32=true')
    $localBin = Join-Path $env:USERPROFILE '.local\bin'
    if (Test-Path $localBin) {
        $env:Path = "$localBin;$env:Path"
    }
    $agent = Find-Agent
}
if (-not $agent) {
    Write-Host 'Open a new PowerShell window, then run:'
    Write-Host "  Set-Location $Root"
    Write-Host '  agent login'
    Write-Host "  agent worker start --name $Name --worker-dir $Root"
    exit 1
}

Write-Host "Repo: $Root"
Write-Host 'Leave the cbu_lab tunnel active. In this PowerShell window:'
Write-Host '  agent login'
Write-Host "  agent worker start --name $Name --worker-dir $Root"
Write-Host 'Then, in Cursor, choose LeonPC-ridcheck under My Machines.'

if ($Start) {
    & $agent login
    & $agent worker start --name $Name --worker-dir $Root
}
