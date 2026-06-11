# Disable JCTVC debug functionality to avoid GCC 15.2 time.h issues
# Debug features aren't needed for BPG encoding

$ErrorActionPreference = "Stop"

Write-Host "Disabling JCTVC Debug functionality..." -ForegroundColor Cyan

$files = @(
    "jctvc\TLibCommon\Debug.cpp",
    "jctvc\TLibCommon\Debug.h"
)

foreach ($file in $files) {
    if (Test-Path $file) {
        $content = Get-Content $file -Raw
        
        # Wrap entire file content in #if 0 ... #endif
        $wrapped = "#if 0 // Disabled for GCC 15.2 compatibility`n" + $content + "`n#endif`n"
        
        Set-Content -Path $file -Value $wrapped -NoNewline
        Write-Host "  ✓ Disabled $file" -ForegroundColor Green
    }
}

# Remove Debug.cpp from compilation in build script
$buildScript = "build_native_lib_with_jctvc.bat"
if (Test-Path $buildScript) {
    $content = Get-Content $buildScript -Raw
    
    # Skip Debug.cpp compilation
    $content = $content -replace '(echo Compiling Debug\.cpp\.\.\..*?if errorlevel 1 goto error)', '    REM Skipping Debug.cpp (disabled)'
    
    Set-Content -Path $buildScript -Value $content -NoNewline
    Write-Host "  ✓ Updated build script to skip Debug.cpp" -ForegroundColor Green
}

Write-Host "`nDebug functionality disabled. Retry build." -ForegroundColor Yellow
