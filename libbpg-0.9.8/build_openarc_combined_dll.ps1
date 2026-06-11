[CmdletBinding()]
param(
    [int]$Jobs = 0,
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'

# In PowerShell 7+, native command stderr can be promoted to errors when
# $ErrorActionPreference is 'Stop'. We compile third-party code (JCTVC/FFmpeg)
# that emits warnings on stderr; those should not abort the build.
$PSNativeCommandUseErrorActionPreference = $false

function Resolve-ToolchainBin {
    $msys64 = 'C:\msys64'
    $candidates = @(
        (Join-Path $msys64 'mingw64\bin'),
        (Join-Path $msys64 'ucrt64\bin'),
        (Join-Path $msys64 'clang64\bin')
    )
    foreach ($dir in $candidates) {
        if (Test-Path (Join-Path $dir 'g++.exe')) {
            return $dir
        }
    }
    return $null
}

function Invoke-ParallelCompile {
    param(
        [Parameter(Mandatory)]
        [string]$Title,
        [Parameter(Mandatory)]
        [object[]]$Tasks,
        [Parameter(Mandatory)]
        [int]$Throttle
    )

    if (-not $Tasks -or $Tasks.Count -eq 0) {
        return
    }

    Write-Host ""
    Write-Host "========================================"
    Write-Host $Title
    Write-Host "========================================"

    $Tasks | ForEach-Object -Parallel {
        $name = $_.Name
        Write-Host "  Compiling $name..."
        $output = & $_.Compiler @($_.Args) 2>&1
        if ($output) {
            $output | ForEach-Object { Write-Host $_ }
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Compile failed: $name"
        }
    } -ThrottleLimit $Throttle
}

function Invoke-ArChunked {
    param(
        [Parameter(Mandatory)]
        [string]$Ar,
        [Parameter(Mandatory)]
        [string]$ArchivePath,
        [Parameter(Mandatory)]
        [string[]]$Objects
    )

    if (Test-Path $ArchivePath) {
        Remove-Item -Force $ArchivePath
    }

    $chunkSize = 200
    for ($i = 0; $i -lt $Objects.Count; $i += $chunkSize) {
        $chunk = $Objects[$i..([Math]::Min($i + $chunkSize - 1, $Objects.Count - 1))]
        $output = & $Ar 'rcs' $ArchivePath @chunk 2>&1
        if ($output) {
            $output | ForEach-Object { Write-Host $_ }
        }
        if ($LASTEXITCODE -ne 0) {
            throw "ar failed while creating $ArchivePath"
        }
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

if ($Jobs -le 0) {
    $Jobs = [Environment]::ProcessorCount
}

Write-Host "========================================"
Write-Host "Building OpenArc BPG DLL (Combined)"
Write-Host "x265 + JCTVC Encoders"
Write-Host "Parallel jobs: $Jobs"
Write-Host "========================================"

$toolBin = Resolve-ToolchainBin
if ($toolBin) {
    $env:PATH = "$toolBin;$env:PATH"
}

$gcc = if ($toolBin -and (Test-Path (Join-Path $toolBin 'gcc.exe'))) { Join-Path $toolBin 'gcc.exe' } else { 'gcc' }
$gpp = if ($toolBin -and (Test-Path (Join-Path $toolBin 'g++.exe'))) { Join-Path $toolBin 'g++.exe' } else { 'g++' }
$ar  = if ($toolBin -and (Test-Path (Join-Path $toolBin 'ar.exe')))  { Join-Path $toolBin 'ar.exe' } else { 'ar' }

if ($Clean -and (Test-Path 'obj_dll')) {
    Remove-Item -Recurse -Force 'obj_dll'
}

if (-not (Test-Path 'obj_dll')) {
    New-Item -ItemType Directory -Force 'obj_dll' | Out-Null
}

$baseCFlags = @(
    '-O3','-Wall','-pipe','-fno-strict-aliasing','-fno-asynchronous-unwind-tables','-fdata-sections','-ffunction-sections',
    '-fno-math-errno','-fno-signed-zeros','-fno-tree-vectorize','-fomit-frame-pointer',
    '-D_FILE_OFFSET_BITS=64','-D_LARGEFILE_SOURCE','-D_REENTRANT'
)

$cFlags = $baseCFlags + @('-std=c99')

# For C++ on MinGW-w64 x64 we need proper unwind tables for SEH exceptions.
$cxxFlags = $baseCFlags + @('-std=c++11','-fexceptions','-funwind-tables','-fasynchronous-unwind-tables')

$includes = @(
    '-I.','-Ilibavutil','-Ilibavcodec',
    '-Ijctvc','-Ijctvc/TLibCommon','-Ijctvc/TLibEncoder','-Ijctvc/TLibVideoIO','-Ijctvc/libmd5',
    '-Ix265'
)

$defines = @(
    '-DUSE_X265','-DUSE_JCTVC','-DHAVE_AV_CONFIG_H=1','-DBPG_ENCODER_LIB',
    '-DCONFIG_BPG_VERSION="0.9.8"','-DCONFIG_WIN32=1','-DFF_MEMORY_POISON=0x2a',
    '-DMSYS_PROJECT','-D_MSYS2','-D_CRT_SECURE_NO_DEPRECATE','-D_CRT_SECURE_NO_WARNINGS',
    '-D_CRT_NONSTDC_NO_WARNINGS','-D_WIN32_WINNT=0x0600','-DBUILDING_DLL',
    '-D_ISOC99_SOURCE','-D_GNU_SOURCE','-DHAVE_STRING_H','-DHAVE_STDINT_H',
    '-DHAVE_INTTYPES_H','-DHAVE_MALLOC_H','-D__STDC_LIMIT_MACROS'
)

$jctvcWarnings = @(
    '-Wno-sign-compare','-Wno-unused-parameter','-Wno-missing-field-initializers',
    '-Wno-misleading-indentation','-Wno-class-memaccess'
)

$jctvcCxxFlags = $cxxFlags + $jctvcWarnings

# --- Compile tasks ---

$jctvcCommon = Get-ChildItem 'jctvc\TLibCommon\*.cpp' | ForEach-Object {
    [pscustomobject]@{
        Name     = "TLibCommon\$($_.Name)"
        Compiler = $gpp
        Args     = $jctvcCxxFlags + $includes + $defines + @('-c', $_.FullName, '-o', (Join-Path 'obj_dll' "TLib_$($_.BaseName).o"))
    }
}

$jctvcEnc = Get-ChildItem 'jctvc\TLibEncoder\*.cpp' | ForEach-Object {
    [pscustomobject]@{
        Name     = "TLibEncoder\$($_.Name)"
        Compiler = $gpp
        Args     = $jctvcCxxFlags + $includes + $defines + @('-c', $_.FullName, '-o', (Join-Path 'obj_dll' "TEnc_$($_.BaseName).o"))
    }
}

$jctvcVid = Get-ChildItem 'jctvc\TLibVideoIO\*.cpp' | ForEach-Object {
    [pscustomobject]@{
        Name     = "TLibVideoIO\$($_.Name)"
        Compiler = $gpp
        Args     = $jctvcCxxFlags + $includes + $defines + @('-c', $_.FullName, '-o', (Join-Path 'obj_dll' "TVid_$($_.BaseName).o"))
    }
}

$md5Tasks = Get-ChildItem 'jctvc\libmd5\*.c' | ForEach-Object {
    [pscustomobject]@{
        Name     = "libmd5\$($_.Name)"
        Compiler = $gcc
        Args     = $cFlags + $includes + $defines + @('-c', $_.FullName, '-o', (Join-Path 'obj_dll' "md5_$($_.BaseName).o"))
    }
}

$topLevelCpp = @('jctvc\TAppEncCfg.cpp','jctvc\TAppEncTop.cpp','jctvc\program_options_lite.cpp') | ForEach-Object {
    $src = $_
    [pscustomobject]@{
        Name     = $src
        Compiler = $gpp
        Args     = $jctvcCxxFlags + $includes + $defines + @('-c', $src, '-o', (Join-Path 'obj_dll' ("{0}.o" -f ([IO.Path]::GetFileNameWithoutExtension($src)))))
    }
}

$coreTasks = @(
    [pscustomobject]@{ Name='jctvc_glue.cpp'; Compiler=$gpp; Args=$jctvcCxxFlags + $includes + $defines + @('-c','jctvc_glue.cpp','-o',(Join-Path 'obj_dll' 'jctvc_glue.o')) },
    [pscustomobject]@{ Name='x265_glue.c';   Compiler=$gcc; Args=$cFlags        + $includes + $defines + @('-c','x265_glue.c','-o',(Join-Path 'obj_dll' 'x265_glue.o')) },
    [pscustomobject]@{ Name='bpgenc.c';      Compiler=$gcc; Args=$cFlags        + $includes + $defines + @('-c','bpgenc.c','-o',(Join-Path 'obj_dll' 'bpgenc.o')) },
    [pscustomobject]@{ Name='libbpg.c';      Compiler=$gcc; Args=$cFlags        + $includes + $defines + @('-c','libbpg.c','-o',(Join-Path 'obj_dll' 'libbpg.o')) }
)

$avutilTasks = Get-ChildItem 'libavutil\*.c' | ForEach-Object {
    [pscustomobject]@{
        Name     = "libavutil\$($_.Name)"
        Compiler = $gcc
        Args     = $cFlags + $includes + $defines + @('-c', $_.FullName, '-o', (Join-Path 'obj_dll' "avutil_$($_.BaseName).o"))
    }
}

$avcodecSrcs = @(
    'cabac.c','golomb.c','hevc.c','hevc_cabac.c','hevc_filter.c','hevc_mvs.c','hevc_ps.c','hevc_refs.c','hevc_sei.c','hevcdsp.c','hevcpred.c','utils.c','videodsp.c'
)

$avcodecTasks = $avcodecSrcs | ForEach-Object {
    $src = Join-Path 'libavcodec' $_
    [pscustomobject]@{
        Name     = "libavcodec\$_"
        Compiler = $gcc
        Args     = $cFlags + $includes + $defines + @('-c', $src, '-o', (Join-Path 'obj_dll' ("avcodec_{0}.o" -f ([IO.Path]::GetFileNameWithoutExtension($_)))))
    }
}

$apiTasks = @(
    [pscustomobject]@{ Name='bpg_api.c';          Compiler=$gcc; Args=$cFlags + $includes + $defines + @('-c','bpg_api.c','-o',(Join-Path 'obj_dll' 'bpg_api.o')) },
    [pscustomobject]@{ Name='openarc_bpg_dll.c';  Compiler=$gcc; Args=$cFlags + $includes + $defines + @('-c','openarc_bpg_dll.c','-o',(Join-Path 'obj_dll' 'openarc_bpg_dll.o')) }
)

Invoke-ParallelCompile -Title 'Step 1: Compiling JCTVC TLibCommon' -Tasks $jctvcCommon -Throttle $Jobs
Invoke-ParallelCompile -Title 'Step 2: Compiling JCTVC TLibEncoder' -Tasks $jctvcEnc -Throttle $Jobs
Invoke-ParallelCompile -Title 'Step 3: Compiling JCTVC TLibVideoIO' -Tasks $jctvcVid -Throttle $Jobs
Invoke-ParallelCompile -Title 'Step 4: Compiling JCTVC libmd5' -Tasks $md5Tasks -Throttle $Jobs
Invoke-ParallelCompile -Title 'Step 5: Compiling JCTVC top-level files' -Tasks $topLevelCpp -Throttle $Jobs
Invoke-ParallelCompile -Title 'Step 6-7: Compiling glue + BPG core' -Tasks $coreTasks -Throttle $Jobs
Invoke-ParallelCompile -Title 'Step 7b: Compiling libavutil (decoder deps)' -Tasks $avutilTasks -Throttle $Jobs
Invoke-ParallelCompile -Title 'Step 7c: Compiling libavcodec (decoder deps)' -Tasks $avcodecTasks -Throttle $Jobs
Invoke-ParallelCompile -Title 'Step 8: Compiling BPG API layer' -Tasks $apiTasks -Throttle $Jobs

Write-Host ""
Write-Host "========================================"
Write-Host "Step 9: Creating combined object archives"
Write-Host "========================================"

# Get all objects
$all_objects = Get-ChildItem 'obj_dll\*.o' | Select-Object -ExpandProperty FullName
if (-not $all_objects -or $all_objects.Count -eq 0) {
    throw 'No object files found in obj_dll'
}

# Create static library for FFI - now includes bpgenc.o (without main() thanks to BPG_ENCODER_LIB)
Write-Host "  Creating libbpg_native.a with $($all_objects.Count) objects (for FFI linking)"
Invoke-ArChunked -Ar $ar -ArchivePath (Join-Path 'obj_dll' 'libbpg_native.a') -Objects $all_objects

# Create combined archive for DLL (same as static lib now that BPG_ENCODER_LIB is defined)
Write-Host "  Creating libbpg_combined.a with $($all_objects.Count) objects (for DLL)"
Invoke-ArChunked -Ar $ar -ArchivePath (Join-Path 'obj_dll' 'libbpg_combined.a') -Objects $all_objects

Write-Host ""
Write-Host "========================================"
Write-Host "Step 10: Linking openarc_bpg.dll"
Write-Host "========================================"

$linkOutput = & $gpp -shared -o 'openarc_bpg.dll' `
    '-Wl,--out-implib,openarc_bpg.lib' `
    '-Wl,--export-all-symbols' `
    '-Wl,--whole-archive' (Join-Path 'obj_dll' 'libbpg_combined.a') '-Wl,--no-whole-archive' `
    '-lx265' '-lpng' '-ljpeg' '-lz' '-lpthread' '-lm' '-lws2_32' 2>&1

if ($linkOutput) {
    $linkOutput | ForEach-Object { Write-Host $_ }
}

if ($LASTEXITCODE -ne 0) {
    throw 'Link failed'
}

Write-Host ""
Write-Host "========================================"
Write-Host "OpenArc BPG DLL Build Complete!"
Write-Host "========================================"

Get-Item 'openarc_bpg.dll' | ForEach-Object { Write-Host ("DLL: openarc_bpg.dll ({0} bytes)" -f $_.Length) }
Get-Item 'openarc_bpg.lib' | ForEach-Object { Write-Host ("Import Lib: openarc_bpg.lib ({0} bytes)" -f $_.Length) }
