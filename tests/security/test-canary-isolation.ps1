
param(
    [Parameter(Mandatory)]
    [string]$UserAKey,

    [Parameter(Mandatory)]
    [string]$UserBKey
)

$ErrorActionPreference = "Stop"

$tests = @(
    @{
        Name = "User A isolation"
        Key = $UserAKey
        Expected = "ALPHA-116"
        Forbidden = "BRAVO-116"
        OwnDocument = "Project Alpha"
        OtherDocument = "Project Bravo"
    },
    @{
        Name = "User B isolation"
        Key = $UserBKey
        Expected = "BRAVO-116"
        Forbidden = "ALPHA-116"
        OwnDocument = "Project Bravo"
        OtherDocument = "Project Alpha"
    }
)

$failures = 0

foreach ($test in $tests) {
    $body = @{
        q = "What is the security project canary marker?"
        notebook = "canary-test-116"
        top_k = 4
    } | ConvertTo-Json

    try {
        $result = Invoke-RestMethod `
            -Uri "http://127.0.0.1:8001/v1/chat" `
            -Method POST `
            -Headers @{"X-API-Key" = $test.Key} `
            -ContentType "application/json" `
            -Body $body

        $answer = [string]$result.answer
        $citations = $result.citations | ConvertTo-Json -Depth 6

        $passed = (
            $result.grounded -eq $true -and
            $result.hits_count -eq 1 -and
            $result.citations.Count -eq 1 -and
            $answer.Contains($test.Expected) -and
            -not $answer.Contains($test.Forbidden) -and
            $citations.Contains($test.OwnDocument) -and
            -not $citations.Contains($test.OtherDocument)
        )

        if ($passed) {
            Write-Host "PASS: $($test.Name)" -ForegroundColor Green
        } else {
            Write-Host "FAIL: $($test.Name)" -ForegroundColor Red
            $failures++
        }
    }
    catch {
        Write-Host "ERROR: $($test.Name)" -ForegroundColor Red
        $failures++
    }
}

if ($failures -gt 0) {
    throw "$failures isolation test(s) failed."
}

Write-Host "All canary isolation tests passed." -ForegroundColor Green
