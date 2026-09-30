param(
    [int]$Port = 2100,
    [string]$CarlaRoot = 'H:\CARLA_0.9.16',
    [int]$ResX = 800,
    [int]$ResY = 600
)

$exePath = Join-Path $CarlaRoot 'CarlaUE4.exe'
if (-not (Test-Path $exePath)) {
    throw "CARLA executable not found at $exePath"
}

$args = @(
    "-carla-rpc-port=$Port",
    '-quality-level=Low',
    '-windowed',
    "-ResX=$ResX",
    "-ResY=$ResY",
    '-nosound'
)

Start-Process -FilePath $exePath -ArgumentList $args