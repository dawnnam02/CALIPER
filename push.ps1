﻿# CALIPER 를 깃허브에 올린다 (PowerShell 판).
#
# 쓰는 법:
#   .\push.ps1            -> 비공개로 생성
#   .\push.ps1 public     -> 공개로 생성
#
# 하는 일:
#   - PATH 를 새로고침해서 gh 를 찾는다 (설치 직후 세션에서도 동작)
#   - 로그인 확인 (안 돼 있으면 안내하고 멈춘다)
#   - pyproject.toml 의 USERNAME 자리를 실제 계정으로 바꾸고 커밋
#   - 저장소를 만들고 푸시. 이미 있으면 원격만 붙여서 푸시

param(
    [ValidateSet("private", "public")]
    [string]$Visibility = "private"
)

# PowerShell 5.1 주의: 네이티브 exe 의 stderr 를 리다이렉트하면 정상 종료(0)여도
# NativeCommandError 로 잡힌다. 그래서 리다이렉트하지 않고 $LASTEXITCODE 만 본다.
$ErrorActionPreference = "Continue"
$RepoName = "CALIPER"

# 한글이 깨지지 않게
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch {}
$OutputEncoding = [Text.Encoding]::UTF8

Set-Location -LiteralPath $PSScriptRoot

# --- 0. PATH 새로고침 -------------------------------------------------------
# winget 으로 방금 설치했으면 이미 열려 있던 세션에는 PATH 가 반영돼 있지 않다.
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
            [Environment]::GetEnvironmentVariable("Path", "User")

$gh = Get-Command gh -ErrorAction SilentlyContinue
if (-not $gh) {
    $fallback = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\GitHub.cli_Microsoft.Winget.Source_8wekyb3d8bbwe\bin\gh.exe"
    if (Test-Path $fallback) {
        $ghExe = $fallback
    } else {
        Write-Error "gh 를 찾을 수 없다. 먼저 설치해라:`n  winget install --id GitHub.cli --scope user"
        exit 1
    }
} else {
    $ghExe = $gh.Source
}
Write-Host "gh : $ghExe" -ForegroundColor DarkGray

# --- 1. 인증 확인 -----------------------------------------------------------
Write-Host "`n== 1. 인증 확인 ==" -ForegroundColor Cyan
& $ghExe auth status *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host @"

로그인이 안 돼 있다. 먼저 이걸 실행해라 (브라우저가 열린다):

    gh auth login

  What account?   GitHub.com
  Protocol?       HTTPS
  Authenticate Git with your GitHub credentials?   Yes
  How to login?   Login with a web browser

8자리 코드를 복사 -> Enter -> 브라우저에 붙여넣기 -> Authorize.
끝나면 이 스크립트를 다시 실행하면 된다.

"@ -ForegroundColor Yellow
    exit 3
}
$login = (& $ghExe api user --jq .login).Trim()
Write-Host "   로그인됨: $login" -ForegroundColor Green

# --- 2. 프로젝트 URL 교체 ---------------------------------------------------
Write-Host "`n== 2. pyproject.toml 의 URL 을 실제 계정으로 ==" -ForegroundColor Cyan
$pyproject = "pyproject.toml"
$content = Get-Content $pyproject -Raw -Encoding UTF8
if ($content -match "USERNAME") {
    $content = $content -replace "https://github\.com/USERNAME/CALIPER",
                                 "https://github.com/$login/CALIPER"
    [System.IO.File]::WriteAllText((Resolve-Path $pyproject), $content,
                                   (New-Object System.Text.UTF8Encoding $false))
    git add $pyproject
    git commit -q -m "Point the project URL at the actual repository"
    Write-Host "   -> https://github.com/$login/CALIPER" -ForegroundColor Green
} else {
    Write-Host "   이미 교체돼 있음" -ForegroundColor DarkGray
}

# --- 3. 저장소 생성 및 푸시 -------------------------------------------------
Write-Host "`n== 3. 저장소 만들고 푸시 ==" -ForegroundColor Cyan
$branch = (git branch --show-current).Trim()

$hasRemote = $false
git remote get-url origin *> $null
if ($LASTEXITCODE -eq 0) { $hasRemote = $true }

if ($hasRemote) {
    Write-Host "   원격이 이미 있다: $(git remote get-url origin)" -ForegroundColor DarkGray
    git push -u origin $branch
} else {
    & $ghExe repo view "$login/$RepoName" *> $null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   깃허브에 $RepoName 이 이미 있다. 원격만 연결한다." -ForegroundColor DarkGray
        git remote add origin "https://github.com/$login/$RepoName.git"
        git push -u origin $branch
    } else {
        $desc = "Budget-aware allocation and calibration for de novo protein " +
                "binder design. A multi-fidelity cascade helps under an equal " +
                "compute budget but not on a fixed pool; the crossover is the finding."
        & $ghExe repo create $RepoName "--$Visibility" --source=. --remote=origin --push --description $desc
    }
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n푸시가 실패했다 (종료코드 $LASTEXITCODE). 위 메시지를 확인해라." -ForegroundColor Red
    exit $LASTEXITCODE
}

# --- 완료 -------------------------------------------------------------------
Write-Host "`n== 완료 ==" -ForegroundColor Green
Write-Host "   https://github.com/$login/$RepoName"
Write-Host "   공개 범위: $Visibility"
if ($Visibility -eq "private") {
    Write-Host "`n   나중에 공개로 바꾸려면:" -ForegroundColor DarkGray
    Write-Host "   gh repo edit --visibility public --accept-visibility-change-consequences" -ForegroundColor DarkGray
}
