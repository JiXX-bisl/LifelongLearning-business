[CmdletBinding()]
param([string]$Python = "python")
$ErrorActionPreference = "Stop"
& $Python -B -X utf8 (Join-Path $PSScriptRoot "run_windows_tests.py") --profile direct
exit $LASTEXITCODE
