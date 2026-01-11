<#
.SYNOPSIS
    LLM Council Test Matrix - Comprehensive testing across scenarios and models

.DESCRIPTION
    Tests the LLM Council CLI with various permutations of:
    - Topics and objectives
    - Number of personas
    - Consensus types
    - Different LLM models (LM Studio local + OpenAI)

.NOTES
    Results are saved to test_results.json
#>

param(
    [string]$LMStudioBase = "http://169.254.83.107:1234/v1",
    [string]$OutputFile = "test_results.json",
    [switch]$SkipOpenAI,
    [switch]$VerboseOutput
)

$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot "venv\Scripts\python.exe"

# Test Scenarios Matrix
$Scenarios = @(
    @{
        Name = "SimpleDecision"
        Topic = "ColorChoice"
        Objective = "Choose red or blue"
        Personas = 3
        MaxRounds = 2
        ConsensusType = "majority"
    },
    @{
        Name = "TechnicalChoice"
        Topic = "FrameworkSelection"
        Objective = "Choose React or Vue"
        Personas = 3
        MaxRounds = 2
        ConsensusType = "majority"
    },
    @{
        Name = "ArchitectureDebate"
        Topic = "SystemDesign"
        Objective = "Monolith or microservices"
        Personas = 3
        MaxRounds = 2
        ConsensusType = "majority"
    }
)

# Models to test
$Models = @(
    @{
        Name = "LMStudio-Qwen3"
        Model = "openai/qwen/qwen3-coder-30b"
        ApiBase = $LMStudioBase
        ApiKey = "lm-studio"
        Type = "local"
    }
)

# Add OpenAI models if not skipped and API key exists
if (-not $SkipOpenAI -and $env:OPENAI_API_KEY) {
    $Models += @(
        @{
            Name = "GPT4o-mini"
            Model = "gpt-4o-mini"
            ApiBase = $null
            ApiKey = $env:OPENAI_API_KEY
            Type = "openai"
        },
        @{
            Name = "GPT4o"
            Model = "gpt-4o"
            ApiBase = $null
            ApiKey = $env:OPENAI_API_KEY
            Type = "openai"
        },
        @{
            Name = "GPT35-Turbo"
            Model = "gpt-3.5-turbo"
            ApiBase = $null
            ApiKey = $env:OPENAI_API_KEY
            Type = "openai"
        }
    )
}

# Results storage
$Results = @{
    TestRun = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    TotalTests = 0
    Passed = 0
    Failed = 0
    Skipped = 0
    Tests = @()
}

function Test-CouncilSession {
    param(
        [hashtable]$Scenario,
        [hashtable]$Model
    )

    $testName = "$($Scenario.Name)_$($Model.Name)"
    Write-Host "`n[TEST] $testName" -ForegroundColor Cyan

    $result = @{
        TestName = $testName
        Scenario = $Scenario.Name
        Model = $Model.Name
        ModelType = $Model.Type
        StartTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Status = "RUNNING"
        Duration = 0
        ConsensusReached = $false
        RoundsCompleted = 0
        Error = $null
    }

    $startTime = Get-Date

    try {
        # Build arguments as a single string with proper quoting
        $argString = "-m llm_council discuss"
        $argString += " --topic `"$($Scenario.Topic)`""
        $argString += " --objective `"$($Scenario.Objective)`""
        $argString += " --personas $($Scenario.Personas)"
        $argString += " --max-rounds $($Scenario.MaxRounds)"
        $argString += " --consensus-type $($Scenario.ConsensusType)"
        $argString += " --output json"
        $argString += " --quiet"
        $argString += " --model `"$($Model.Model)`""

        if ($Model.ApiBase) {
            $argString += " --api-base `"$($Model.ApiBase)`""
        }

        if ($Model.ApiKey) {
            $argString += " --api-key `"$($Model.ApiKey)`""
        }

        $arguments = $argString

        # Run using Start-Process to capture output properly
        $tempOut = [System.IO.Path]::GetTempFileName()
        $tempErr = [System.IO.Path]::GetTempFileName()

        $proc = Start-Process -FilePath $PythonPath -ArgumentList $arguments -WorkingDirectory $ProjectRoot -NoNewWindow -Wait -PassThru -RedirectStandardOutput $tempOut -RedirectStandardError $tempErr

        $stdout = Get-Content $tempOut -Raw -ErrorAction SilentlyContinue
        $stderr = Get-Content $tempErr -Raw -ErrorAction SilentlyContinue
        Remove-Item $tempOut, $tempErr -ErrorAction SilentlyContinue

        $endTime = Get-Date
        $result.Duration = [math]::Round(($endTime - $startTime).TotalSeconds, 2)
        $result.EndTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

        # Check for valid JSON output
        if ($stdout -and $stdout.Contains('"topic"')) {
            try {
                $jsonStart = $stdout.IndexOf("{")
                if ($jsonStart -ge 0) {
                    $jsonContent = $stdout.Substring($jsonStart)
                    $parsed = $jsonContent | ConvertFrom-Json

                    $result.ConsensusReached = $parsed.consensus_reached
                    $result.RoundsCompleted = $parsed.rounds.Count
                    if ($parsed.final_consensus) {
                        $result.FinalConsensus = $parsed.final_consensus.Substring(0, [Math]::Min(200, $parsed.final_consensus.Length))
                    }
                    $result.Status = "PASS"
                    Write-Host "  [PASS] Consensus: $($result.ConsensusReached), Rounds: $($result.RoundsCompleted), Duration: $($result.Duration)s" -ForegroundColor Green
                }
            }
            catch {
                $result.Status = "FAIL"
                $result.Error = "JSON parse error: $_"
                Write-Host "  [FAIL] JSON parse error" -ForegroundColor Red
            }
        }
        elseif ($proc.ExitCode -eq 0) {
            $result.Status = "PASS"
            $result.Error = "No JSON but exit 0"
            Write-Host "  [PASS] Completed (no JSON)" -ForegroundColor Green
        }
        else {
            $result.Status = "FAIL"
            $result.Error = if ($stderr) { $stderr.Substring(0, [Math]::Min(300, $stderr.Length)) } else { "Exit code: $($proc.ExitCode)" }
            Write-Host "  [FAIL] $($result.Error)" -ForegroundColor Red
        }
    }
    catch {
        $result.Status = "ERROR"
        $result.Error = $_.Exception.Message
        $result.Duration = [math]::Round(((Get-Date) - $startTime).TotalSeconds, 2)
        Write-Host "  [ERROR] $($_.Exception.Message)" -ForegroundColor Red
    }

    return $result
}

function Test-ModelConnection {
    param([hashtable]$Model)

    Write-Host "[CHECK] $($Model.Name)..." -NoNewline

    $arguments = @("-m", "llm_council", "test-connection", "--model", $Model.Model)
    if ($Model.ApiBase) {
        $arguments += "--api-base"
        $arguments += $Model.ApiBase
    }
    if ($Model.ApiKey) {
        $arguments += "--api-key"
        $arguments += $Model.ApiKey
    }

    try {
        $tempOut = [System.IO.Path]::GetTempFileName()
        $tempErr = [System.IO.Path]::GetTempFileName()
        $proc = Start-Process -FilePath $PythonPath -ArgumentList $arguments -WorkingDirectory $ProjectRoot -NoNewWindow -Wait -PassThru -RedirectStandardOutput $tempOut -RedirectStandardError $tempErr
        $output = Get-Content $tempOut -Raw -ErrorAction SilentlyContinue
        Remove-Item $tempOut, $tempErr -ErrorAction SilentlyContinue

        if ($proc.ExitCode -eq 0) {
            Write-Host " OK" -ForegroundColor Green
            return $true
        }
        else {
            Write-Host " SKIP" -ForegroundColor Yellow
            return $false
        }
    }
    catch {
        Write-Host " ERROR" -ForegroundColor Red
        return $false
    }
}

# Main execution
Write-Host "========================================" -ForegroundColor White
Write-Host "  LLM Council Test Matrix" -ForegroundColor White
Write-Host "========================================" -ForegroundColor White
Write-Host "Scenarios: $($Scenarios.Count)"
Write-Host "Models: $($Models.Count)"
Write-Host ""

# Test model connections
Write-Host "Testing model connections:" -ForegroundColor Cyan
$AvailableModels = @()
foreach ($model in $Models) {
    if (Test-ModelConnection -Model $model) {
        $AvailableModels += $model
    }
}

if ($AvailableModels.Count -eq 0) {
    Write-Host "`n[ERROR] No models available!" -ForegroundColor Red
    exit 1
}

Write-Host "`nRunning $($Scenarios.Count * $AvailableModels.Count) tests..." -ForegroundColor Cyan

# Run test matrix
foreach ($scenario in $Scenarios) {
    foreach ($model in $AvailableModels) {
        $Results.TotalTests++
        $testResult = Test-CouncilSession -Scenario $scenario -Model $model
        $Results.Tests += $testResult

        switch ($testResult.Status) {
            "PASS" { $Results.Passed++ }
            "FAIL" { $Results.Failed++ }
            "ERROR" { $Results.Failed++ }
            default { $Results.Skipped++ }
        }
    }
}

# Summary
Write-Host "`n========================================" -ForegroundColor White
Write-Host "  Results Summary" -ForegroundColor White
Write-Host "========================================" -ForegroundColor White
Write-Host "Total: $($Results.TotalTests) | Passed: $($Results.Passed) | Failed: $($Results.Failed)"

# Save results
$OutputPath = Join-Path $ProjectRoot $OutputFile
$Results | ConvertTo-Json -Depth 10 | Set-Content -Path $OutputPath -Encoding UTF8
Write-Host "Saved to: $OutputPath"

if ($Results.Passed -gt 0) {
    Write-Host "`n[SUCCESS] $($Results.Passed) tests passed!" -ForegroundColor Green
    exit 0
}
else {
    Write-Host "`n[FAILURE] No tests passed!" -ForegroundColor Red
    exit 1
}
