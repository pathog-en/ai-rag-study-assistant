param(
  [Parameter(Mandatory=$true)] [string] $BaseUrl,
  [Parameter(Mandatory=$true)] [string] $AdminKey
)

$ErrorActionPreference = "Stop"

function Assert-True($Condition, $Message) {
  if (-not $Condition) { throw "ASSERT FAILED: $Message" }
}

function Post-Json($Url, $Headers, $BodyObj) {
  $json = $BodyObj | ConvertTo-Json -Depth 10
  return Invoke-RestMethod -Method POST -Uri $Url -Headers $Headers -ContentType "application/json" -Body $json
}

Write-Host "== RAG Study Assistant Happy Path Test ==" -ForegroundColor Cyan
Write-Host "BaseUrl: $BaseUrl"

# 1) Create User A key
Write-Host "`n[1] Create User A key" -ForegroundColor Yellow
$a = Invoke-RestMethod -Method POST -Uri "$BaseUrl/admin/create_user_key" -Headers @{ "x-admin-key" = $AdminKey } -ContentType "application/json"
Assert-True ($null -ne $a.api_key) "User A api_key missing"
Assert-True ($null -ne $a.user_id)  "User A user_id missing"
$userKeyA = $a.api_key
Write-Host "User A: $($a.user_id)"

# 2) Create User B key (for isolation test)
Write-Host "`n[2] Create User B key (isolation check)" -ForegroundColor Yellow
$b = Invoke-RestMethod -Method POST -Uri "$BaseUrl/admin/create_user_key" -Headers @{ "x-admin-key" = $AdminKey } -ContentType "application/json"
Assert-True ($null -ne $b.api_key) "User B api_key missing"
Assert-True ($null -ne $b.user_id)  "User B user_id missing"
$userKeyB = $b.api_key
Write-Host "User B: $($b.user_id)"

# 3) /v1/me for A
Write-Host "`n[3] GET /v1/me for User A" -ForegroundColor Yellow
$meA = Invoke-RestMethod -Method GET -Uri "$BaseUrl/v1/me" -Headers @{ "x-api-key" = $userKeyA }
Assert-True ($meA.user_id -eq $a.user_id) "User A /v1/me user_id mismatch"
Write-Host "OK: /v1/me matches User A"

# 4) Ingest notes for A
$notebook = "history"
Write-Host "`n[4] Ingest notes for User A notebook '$notebook'" -ForegroundColor Yellow

$noteA1 = "The Treaty of Versailles ended World War I in 1919. It imposed heavy reparations on Germany."
$noteA2 = "The League of Nations was formed after WWI to prevent future conflicts, but it lacked enforcement power."
$ingA1 = Post-Json "$BaseUrl/v1/ingest" @{ "x-api-key" = $userKeyA } @{ notebook=$notebook; text=$noteA1 }
$ingA2 = Post-Json "$BaseUrl/v1/ingest" @{ "x-api-key" = $userKeyA } @{ notebook=$notebook; text=$noteA2 }

# We don't know your exact response shape; check for at least some non-null response
Assert-True ($null -ne $ingA1) "Ingest A1 returned null"
Assert-True ($null -ne $ingA2) "Ingest A2 returned null"
Write-Host "OK: Ingest returned responses"

# 5) Query A
Write-Host "`n[5] Query User A notebook '$notebook'" -ForegroundColor Yellow
$q1 = Post-Json "$BaseUrl/v1/chat" @{ "x-api-key" = $userKeyA } @{ notebook=$notebook; question="What ended World War I and when?" }

# Try to find an answer field in common places
$answerText =
  $(if ($q1.answer) { $q1.answer }
    elseif ($q1.response) { $q1.response }
    elseif ($q1.message) { $q1.message }
    else { $q1 | ConvertTo-Json -Depth 10 })

Write-Host "`nModel output:"
Write-Host $answerText

# Soft assertions: must mention Versailles and 1919 somewhere
Assert-True ($answerText -match "Versailles") "Answer did not mention Versailles"
Assert-True ($answerText -match "1919") "Answer did not mention 1919"
Write-Host "OK: Retrieved expected facts from ingested notes"

# 6) Tenant isolation test:
# Ingest a unique secret into User B and verify User A can't retrieve it.
Write-Host "`n[6] Tenant isolation test (A should not see B)" -ForegroundColor Yellow
$secret = "B-ONLY-SECRET-" + ([Guid]::NewGuid().ToString("N")).Substring(0,8)
$ingB = Post-Json "$BaseUrl/v1/ingest" @{ "x-api-key" = $userKeyB } @{ notebook=$notebook; text="This is private to user B: $secret" }
Assert-True ($null -ne $ingB) "Ingest B returned null"

$qIso = Post-Json "$BaseUrl/v1/chat" @{ "x-api-key" = $userKeyA } @{ notebook=$notebook; question="What is the private secret for user B?" }

$isoText =
  $(if ($qIso.answer) { $qIso.answer }
    elseif ($qIso.response) { $qIso.response }
    elseif ($qIso.message) { $qIso.message }
    else { $qIso | ConvertTo-Json -Depth 10 })

Write-Host "`nIsolation query output:"
Write-Host $isoText

Assert-True (-not ($isoText -match [Regex]::Escape($secret))) "Isolation FAILED: User A saw User B secret"
Write-Host "OK: Isolation passed (User A did not see User B content)"

Write-Host "`n✅ ALL TESTS PASSED" -ForegroundColor Green