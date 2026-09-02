# 발 추출 마법사 exe 빌드 스크립트.
#
# 이 폴더는 foot_deform_engine 레포와 완전히 독립적이다(2026-09-01, 사용자
# 요청으로 소스까지 통째로 복사해둠 -- 레포 경로를 --paths로 가리키던 이전
# 버전은 두 폴더가 서로 묶여있었어서 폐기). 레포의 webapp/app.py나
# src/foot_engine/을 고쳤으면 이 폴더의 같은 파일도 수동으로 다시 복사해
# 맞출 것 -- 자동 동기화 없음.
#
# 유일하게 안 가진 것: 파이썬 실행 환경(.venv) -- 이건 빌드 "도구"일 뿐
# exe 결과물엔 안 들어가서, 편의상 foot_deform_engine의 .venv를 빌려 쓴다.
# 완전히 독립된 venv가 필요하면 README.md 참고해서 이 폴더에 새로 만들 것.
#
# 실행: 이 폴더에서 `powershell -File build.ps1` (경로는 전부 절대경로로
# 씀 -- --specpath를 옮기면 상대경로 --add-data가 깨진다).

$Root = $PSScriptRoot

$Venv = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Venv)) {
    $Venv = "C:\Users\cani0\foot_deform_engine\.venv\Scripts\python.exe"
}

Push-Location (Join-Path $Root "webapp")
& $Venv -m PyInstaller `
  --name FootExtractWizard --onedir --noconfirm `
  --distpath (Join-Path $Root "dist") `
  --workpath (Join-Path $Root "build") `
  --specpath $Root `
  --add-data "$(Join-Path $Root 'webapp\templates');templates" `
  --add-data "$(Join-Path $Root 'data\models\selfie_multiclass_256x256.tflite');data\models" `
  --hidden-import app `
  --hidden-import stage1_orientation_worker `
  --collect-all mediapipe `
  --exclude-module torch `
  --exclude-module torchvision `
  --exclude-module rembg `
  --exclude-module pymatting `
  --exclude-module onnxruntime `
  --paths (Join-Path $Root "webapp") `
  --paths (Join-Path $Root "src") `
  launcher.py
Pop-Location

Write-Host "빌드 결과: $Root\dist\FootExtractWizard\FootExtractWizard.exe"
