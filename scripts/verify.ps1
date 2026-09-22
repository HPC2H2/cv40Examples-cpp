param(
    [string]$Executable,
    [string]$Models,
    [string]$Baseline,
    [switch]$SkipModels
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
if (!$Executable) { $Executable = Join-Path $projectRoot 'build\cv40.exe' }
if (!$Models) { $Models = Join-Path $projectRoot 'models' }
$Executable = (Resolve-Path -LiteralPath $Executable).Path
$runName = 'verification-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 6)
$runRoot = Join-Path $projectRoot ('output\' + $runName)
$imageRoot = Join-Path $runRoot 'images'
$logRoot = Join-Path $runRoot 'logs'
New-Item -ItemType Directory -Path $imageRoot, $logRoot -Force | Out-Null
$results = New-Object 'System.Collections.Generic.List[object]'
$utf8 = New-Object System.Text.UTF8Encoding($false)

function Add-Result([string]$Name, [string]$Status, [string]$Detail) {
    $results.Add([pscustomobject]@{ name = $Name; status = $Status; detail = $Detail })
    Write-Host "$Status $Name $Detail"
}

function Get-ImageHash([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try {
        return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-', '')
    } finally {
        $algorithm.Dispose()
        $stream.Dispose()
    }
}

function Invoke-Case {
    param([string]$Name, [string[]]$Arguments, [int]$ExpectedExit = 0, [string[]]$Patterns = @())
    $start = New-Object System.Diagnostics.ProcessStartInfo
    $start.FileName = $Executable
    $start.WorkingDirectory = $projectRoot
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $start.StandardOutputEncoding = $utf8
    $start.StandardErrorEncoding = $utf8
    # Apply Windows argv quoting, including quotes and trailing backslashes.
    $start.Arguments = ($Arguments | ForEach-Object {
        '"' + ($_ -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
    }) -join ' '
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $start
    try {
        [void]$process.Start()
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        if (!$process.WaitForExit(180000)) {
            $process.Kill()
            $process.WaitForExit()
            Add-Result $Name 'FAIL' 'Exceeded 180 second timeout'
            return ''
        }
        $text = $stdout.Result + $stderr.Result
        [IO.File]::WriteAllText((Join-Path $logRoot ($Name + '.log')), $text, $utf8)
        $valid = $process.ExitCode -eq $ExpectedExit
        foreach ($pattern in $Patterns) { $valid = $valid -and ($text -match $pattern) }
        if ($valid) { Add-Result $Name 'PASS' "exit=$($process.ExitCode)" }
        else { Add-Result $Name 'FAIL' "exit=$($process.ExitCode), expected=$ExpectedExit; inspect log" }
        return $text
    } finally {
        $process.Dispose()
    }
}

try {
    $listing = Invoke-Case 'registry' @('--list')
    $entryCount = @($listing -split "`r?`n" | Where-Object { $_ -match '^\d+\.\d+(?:-\d+)?\s' }).Count
    if ($entryCount -eq 184) { Add-Result 'entry-count' 'PASS' '184 entries' }
    else { Add-Result 'entry-count' 'FAIL' "$entryCount entries; expected 184" }
    [void](Invoke-Case 'self-test' @('--self-test') -Patterns @('PASS self-test: Sudoku', 'PASS self-test: Unicode'))
    [void](Invoke-Case 'basic-164' @('--all-safe', '--output', $imageRoot) -Patterns @(
        'passed=164 failed=0 skipped=20', '91\.76%', '93\.06%',
        'minutiae=52 descriptor=\[208 x 52\] comparison score=0\.777817', 'solution='))

    # A fixed RNG seed alone does not guarantee stable parallel HOG grouping.
    # Re-run the chapter independently and compare both detector output images.
    foreach ($attempt in 1..3) {
        $repeatRoot = Join-Path $runRoot ('hog-repeat-' + $attempt)
        [void](Invoke-Case ('hog-repeat-' + $attempt) @('--run', '19.5', '--headless', '--output', $repeatRoot) -Patterns @('passed=1 failed=0'))
        $stable = $true
        foreach ($image in @('1_without_meanshift.png', '2_meanshift.png')) {
            $expected = Join-Path $imageRoot ('19.5\' + $image)
            $actual = Join-Path $repeatRoot ('19.5\' + $image)
            $stable = $stable -and ((Get-ImageHash $expected) -eq (Get-ImageHash $actual))
        }
        if ($stable) { Add-Result ('hog-stability-' + $attempt) 'PASS' 'Both images unchanged' }
        else { Add-Result ('hog-stability-' + $attempt) 'FAIL' 'HOG results changed between runs' }
    }

    $modelCases = @(
        @{ id = '24.3'; files = @('MobileNetSSD_deploy.caffemodel', 'MobileNetSSD_deploy.prototxt.txt'); checks = @('detection=bicycle confidence=0\.99', 'detection=dog confidence=0\.99') },
        @{ id = '27.2'; files = @('shape_predictor_68_face_landmarks.dat'); checks = @() },
        @{ id = '27.3'; files = @('shape_predictor_68_face_landmarks.dat'); checks = @() },
        @{ id = '27.4'; files = @('shape_predictor_68_face_landmarks.dat'); checks = @() },
        @{ id = '27.5'; files = @('mmod_human_face_detector.dat'); checks = @('faces=8') },
        @{ id = '28.3'; files = @('shape_predictor_68_face_landmarks.dat'); checks = @() },
        @{ id = '28.3-1'; files = @('shape_predictor_68_face_landmarks.dat'); checks = @() }
    )
    foreach ($case in $modelCases) {
        $missing = @($case.files | Where-Object { !(Test-Path -LiteralPath (Join-Path $Models $_)) })
        if ($SkipModels -or $missing.Count) {
            Add-Result ('model-' + $case.id) 'SKIP' ('Model disabled or absent: ' + ($missing -join ', '))
            continue
        }
        [void](Invoke-Case ('model-' + $case.id) @('--run', $case.id, '--models', $Models, '--headless', '--output', $imageRoot) -Patterns (@('passed=1 failed=0') + $case.checks))
    }

    # Two prerecorded images exercise the real frame callbacks without opening a camera.
    $frameRoot = Join-Path $runRoot 'frames'
    New-Item -ItemType Directory -Path $frameRoot | Out-Null
    $faceSample = Get-ChildItem -LiteralPath (Join-Path $projectRoot 'data') -Recurse -File -Filter image2.jpg |
        Where-Object { $_.Directory.Name -eq 'person' } | Select-Object -First 1
    if (!$faceSample) { throw 'Missing face replay fixture' }
    Copy-Item -LiteralPath $faceSample.FullName -Destination (Join-Path $frameRoot '01.jpg')
    Copy-Item -LiteralPath $faceSample.FullName -Destination (Join-Path $frameRoot '02.jpg')
    foreach ($id in @('8.5', '10.12', '25.2', '27.1-1', '28.1', '28.2')) {
        if ($id -in @('28.1', '28.2') -and ($SkipModels -or !(Test-Path -LiteralPath (Join-Path $Models 'shape_predictor_68_face_landmarks.dat')))) {
            Add-Result ('replay-' + $id) 'SKIP' 'Landmark model unavailable or disabled'
            continue
        }
        [void](Invoke-Case ('replay-' + $id) @('--run', $id, '--models', $Models, '--headless', '--frames', $frameRoot, '--max-frames', '2', '--output', (Join-Path $runRoot 'replay')) -Patterns @('frames=2 source=replay', 'passed=1 failed=0'))
    }

    $badArguments = @(
        @{ name = 'negative-frames'; args = @('--run', '8.5', '--headless', '--max-frames', '-1'); message = 'nonnegative integer' },
        @{ name = 'zero-frames'; args = @('--run', '8.5', '--headless', '--max-frames', '0'); message = 'greater than zero' },
        @{ name = 'partial-number'; args = @('--run', '3.4', '--seed', '42oops'); message = 'nonnegative integer' },
        @{ name = 'overflow'; args = @('--run', '3.4', '--seed', '9999999999999999999999'); message = 'nonnegative integer' },
        @{ name = 'bad-input'; args = @('--run', '2.1', '--input', '1,broken'); message = 'Invalid --input' },
        @{ name = 'empty-input'; args = @('--run', '2.1', '--input', '1,,2'); message = 'Invalid --input' },
        @{ name = 'conflicting-actions'; args = @('--all-safe', '--run', '3.4'); message = 'Select only one' },
        @{ name = 'unknown-id'; args = @('--run', '999.1', '--headless'); message = 'Unknown example ID' },
        @{ name = 'missing-value'; args = @('--run'); message = 'Missing value' },
        @{ name = 'missing-frames'; args = @('--run', '8.5', '--headless', '--frames', (Join-Path $runRoot 'absent')); message = 'No replay images' }
    )
    foreach ($case in $badArguments) {
        [void](Invoke-Case $case.name $case.args -ExpectedExit 1 -Patterns @($case.message))
    }
    $emptyData = Join-Path $runRoot 'empty-data'
    New-Item -ItemType Directory -Path $emptyData | Out-Null
    foreach ($id in @('24.1', '24.2', '24.4', '24.5', '24.6')) {
        [void](Invoke-Case ('missing-model-' + $id) @('--run', $id, '--headless', '--data', $emptyData, '--models', $emptyData) -ExpectedExit 1 -Patterns @('Missing resource:'))
    }

    if ($Baseline) {
        $items = Get-Content -LiteralPath $Baseline -Raw | ConvertFrom-Json
        $mismatches = @()
        foreach ($item in $items) {
            $path = Join-Path $imageRoot $item.path
            if (!(Test-Path -LiteralPath $path) -or (Get-ImageHash $path) -ne $item.sha256) {
                $mismatches += $item.path
            }
        }
        if ($mismatches.Count) { Add-Result 'image-baseline' 'FAIL' ($mismatches -join ', ') }
        else { Add-Result 'image-baseline' 'PASS' "$($items.Count) PNG hashes unchanged" }
    }
} catch {
    Add-Result 'verification-script' 'FAIL' $_.Exception.Message
} finally {
    $report = [pscustomobject]@{
        timestamp = (Get-Date -Format o)
        executable = $Executable
        images = $imageRoot
        results = @($results.ToArray())
        passed = @($results | Where-Object status -eq 'PASS').Count
        failed = @($results | Where-Object status -eq 'FAIL').Count
        skipped = @($results | Where-Object status -eq 'SKIP').Count
    }
    $json = $report | ConvertTo-Json -Depth 6
    [IO.File]::WriteAllText((Join-Path $runRoot 'report.json'), $json, $utf8)
    [IO.File]::WriteAllText((Join-Path $projectRoot 'output\latest-verification.json'), $json, $utf8)
    Write-Host "Report: $runRoot\report.json"
}
if ($report.failed) { exit 1 }
exit 0
