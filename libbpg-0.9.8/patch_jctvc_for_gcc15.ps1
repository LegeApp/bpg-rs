# Patch JCTVC headers to work with MinGW64 GCC 15.2
# Removes problematic #include <iomanip> that triggers broken ctime/time.h

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Patching JCTVC Headers for GCC 15.2" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$jctvcRoot = "jctvc"
$patchCount = 0

# Function to patch a single file
function Patch-HeaderFile {
    param(
        [string]$FilePath
    )
    
    if (-not (Test-Path $FilePath)) {
        return $false
    }
    
    $content = Get-Content $FilePath -Raw
    $originalContent = $content
    $changed = $false
    
    # Remove #include <iomanip> (causes ctime to be included which triggers time.h bug)
    if ($content -match '#include\s+<iomanip>') {
        Write-Host "  Removing <iomanip> from $FilePath" -ForegroundColor Yellow
        $content = $content -replace '#include\s+<iomanip>\s*\n', ''
        $changed = $true
    }
    
    # Remove mingw_time_fix.h include (we're taking different approach)
    if ($content -match '#include\s+"\.\.\/mingw_time_fix\.h"') {
        Write-Host "  Removing mingw_time_fix.h from $FilePath" -ForegroundColor Yellow
        $content = $content -replace '#include\s+"\.\.\/mingw_time_fix\.h"\s*\n', ''
        $changed = $true
    }
    
    # Remove the old time.h workaround comments
    if ($content -match '//.*Workaround for GCC 15\.x.*\n') {
        $content = $content -replace '//.*Workaround for GCC 15\.x.*\n', ''
        $changed = $true
    }
    
    if ($content -match '//.*Include time\.h compatibility fix.*\n') {
        $content = $content -replace '//.*Include time\.h compatibility fix.*\n', ''
        $changed = $true
    }
    
    # If file uses std::setw, std::setfill, etc., we need full iomanip
    # Instead of removing it, we'll add it back with a workaround
    if ($changed -and ($content -match 'std::setw|std::setfill|std::setprecision')) {
        Write-Host "  File needs iomanip, re-adding with workaround" -ForegroundColor Green
        
        # Find #ifndef at start of file
        if ($content -match '(#ifndef\s+\w+\s*\n#define\s+\w+\s*\n)') {
            $headerGuard = $matches[1]
            # Add time.h include and iomanip after header guard
            $replacement = $headerGuard + @"

// GCC 15.2 MinGW64 time.h workaround - include before C++ headers
#if defined(__MINGW64__) || defined(__MINGW32__)
#define _GLIBCXX_USE_C99 1
extern "C" {
#include <time.h>
}
#endif

"@
            $content = $content -replace [regex]::Escape($headerGuard), $replacement
            
            # Now find last #include and add iomanip after it
            $lines = $content -split "`n"
            $lastIncludeIdx = -1
            for ($i = 0; $i -lt $lines.Count; $i++) {
                if ($lines[$i] -match '^\s*#include' -and $lines[$i] -notmatch 'iomanip') {
                    $lastIncludeIdx = $i
                }
            }
            
            if ($lastIncludeIdx -ge 0) {
                $lines[$lastIncludeIdx] = $lines[$lastIncludeIdx] + "`n#include <iomanip>"
                $content = $lines -join "`n"
            }
            
            Set-Content -Path $FilePath -Value $content -NoNewline
            return $true
        }
    }
    
    if ($changed) {
        Set-Content -Path $FilePath -Value $content -NoNewline
        return $true
    }
    
    return $false
}

# Patch Debug.h specifically
Write-Host "Patching Debug.h..." -ForegroundColor Green
$debugH = Join-Path $jctvcRoot "TLibCommon\Debug.h"
if (Patch-HeaderFile $debugH) {
    $patchCount++
    Write-Host "  ✓ Patched $debugH" -ForegroundColor Green
}

# Patch TEncTop.h (often uses iomanip)
Write-Host "`nPatching TEncTop.h..." -ForegroundColor Green
$tencTopH = Join-Path $jctvcRoot "TEncTop.h"
if (Test-Path $tencTopH) {
    if (Patch-HeaderFile $tencTopH) {
        $patchCount++
        Write-Host "  ✓ Patched $tencTopH" -ForegroundColor Green
    }
}

# Find and patch all .h and .cpp files in JCTVC that include iomanip
Write-Host "`nScanning all JCTVC headers and sources..." -ForegroundColor Green

$allFiles = Get-ChildItem -Path $jctvcRoot -Recurse -Include *.h,*.cpp

foreach ($file in $allFiles) {
    $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
    if ($content -and ($content -match '#include\s+<iomanip>')) {
        Write-Host "  Found iomanip in: $($file.FullName)" -ForegroundColor Yellow
        if (Patch-HeaderFile $file.FullName) {
            $patchCount++
            Write-Host "    ✓ Patched" -ForegroundColor Green
        }
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Patching Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Files patched: $patchCount" -ForegroundColor Green
Write-Host ""
Write-Host "Now run: .\build_native_lib_with_jctvc.bat" -ForegroundColor Yellow
Write-Host ""
