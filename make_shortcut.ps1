# LeafScan Lab - 바탕화면 바로가기 생성
# 실행: 바탕화면_아이콘_만들기.bat 을 더블클릭 (직접 실행해도 됩니다)

$base = $PSScriptRoot
if (-not $base) { $base = (Get-Location).Path }

$py = (Get-Command pythonw.exe -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = (Get-Command python.exe -ErrorAction SilentlyContinue).Source }

if (-not $py) {
    Write-Host ""
    Write-Host "  [실패] 파이썬을 찾을 수 없습니다." -ForegroundColor Red
    Write-Host "  python.org 에서 설치할 때 'Add Python to PATH' 를 체크해야 합니다."
    exit 1
}

$target = Join-Path $base "lab_gui.py"
if (-not (Test-Path $target)) {
    Write-Host ""
    Write-Host "  [실패] lab_gui.py 가 이 폴더에 없습니다:" -ForegroundColor Red
    Write-Host "  $base"
    exit 1
}

$desktop = [Environment]::GetFolderPath('Desktop')
$link    = Join-Path $desktop "LeafScan Lab.lnk"

$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($link)
$sc.TargetPath       = $py
$sc.Arguments        = '"' + $target + '"'
$sc.WorkingDirectory = $base
$sc.Description      = "LeafScan Lab - CNN 실험 콘솔"

$icon = Join-Path $base "leafscan.ico"
if (Test-Path $icon) { $sc.IconLocation = $icon }

$sc.Save()

Write-Host ""
Write-Host "  [완료] 바탕화면에 아이콘을 만들었습니다." -ForegroundColor Green
Write-Host "  사용한 파이썬 : $py"
Write-Host "  바로가기 위치 : $link"
Write-Host ""
Write-Host "  작업 표시줄에 고정하려면 아이콘 우클릭 > '작업 표시줄에 고정' 을 선택하세요."
