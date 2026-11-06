[CmdletBinding()]
param([string]$Python = "python")
$ErrorActionPreference = "Stop"
& $Python -B -X utf8 (Join-Path $PSScriptRoot "../../src/cli/windows_tests.py") --profile values
exit $LASTEXITCODE
