[CmdletBinding()]
param(
    [string]$RunDir,
    [string]$Config,
    [int]$CaseLimit,
    [string]$Python = "python"
)
$ErrorActionPreference = "Stop"
$arguments = @("-B", "-X", "utf8", (Join-Path $PSScriptRoot "run_windows_validation.py"), "--profile", "coverage")
if ($PSBoundParameters.ContainsKey("RunDir")) { $arguments += @("--run-dir", $RunDir) }
if ($PSBoundParameters.ContainsKey("Config")) { $arguments += @("--config", $Config) }
if ($PSBoundParameters.ContainsKey("CaseLimit")) { $arguments += @("--case-limit", "$CaseLimit") }
& $Python @arguments
exit $LASTEXITCODE
