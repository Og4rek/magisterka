[CmdletBinding()]
param(
    [string]$Destination = "",
    [switch]$KeepArchives
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($Destination)) {
    $Destination = Join-Path $PSScriptRoot "..\data\pcam"
}
$Destination = [System.IO.Path]::GetFullPath($Destination)
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))

if (-not $Destination.StartsWith($projectRoot + [System.IO.Path]::DirectorySeparatorChar)) {
    throw "Destination must stay inside the project directory: $Destination"
}

New-Item -ItemType Directory -Path $Destination -Force | Out-Null

# The Zenodo mirror has validation and test filenames swapped relative to the
# checksums in the canonical PCam GitHub README. SourceName identifies the
# Zenodo object; TargetName restores the official benchmark split semantics.
$files = @(
    @{
        SourceName = "camelyonpatch_level_2_split_train_x.h5.gz"
        TargetName = "camelyonpatch_level_2_split_train_x.h5"
        Md5 = "1571f514728f59376b705fc836ff4b63"
        Compressed = $true
    },
    @{
        SourceName = "camelyonpatch_level_2_split_train_y.h5.gz"
        TargetName = "camelyonpatch_level_2_split_train_y.h5"
        Md5 = "35c2d7259d906cfc8143347bb8e05be7"
        Compressed = $true
    },
    @{
        SourceName = "camelyonpatch_level_2_split_test_x.h5.gz"
        TargetName = "camelyonpatch_level_2_split_valid_x.h5"
        Md5 = "d8c2d60d490dbd479f8199bdfa0cf6ec"
        Compressed = $true
    },
    @{
        SourceName = "camelyonpatch_level_2_split_test_y.h5.gz"
        TargetName = "camelyonpatch_level_2_split_valid_y.h5"
        Md5 = "60a7035772fbdb7f34eb86d4420cf66a"
        Compressed = $true
    },
    @{
        SourceName = "camelyonpatch_level_2_split_valid_x.h5.gz"
        TargetName = "camelyonpatch_level_2_split_test_x.h5"
        Md5 = "d5b63470df7cfa627aeec8b9dc0c066e"
        Compressed = $true
    },
    @{
        SourceName = "camelyonpatch_level_2_split_valid_y.h5.gz"
        TargetName = "camelyonpatch_level_2_split_test_y.h5"
        Md5 = "2b85f58b927af9964a4c15b8f7e8f179"
        Compressed = $true
    },
    @{
        SourceName = "camelyonpatch_level_2_split_train_meta.csv"
        TargetName = "camelyonpatch_level_2_split_train_meta.csv"
        Md5 = "5a3dd671e465cfd74b5b822125e65b0a"
        Compressed = $false
    },
    @{
        SourceName = "camelyonpatch_level_2_split_test_meta.csv"
        TargetName = "camelyonpatch_level_2_split_valid_meta.csv"
        Md5 = "3455fd69135b66734e1008f3af684566"
        Compressed = $false
    },
    @{
        SourceName = "camelyonpatch_level_2_split_valid_meta.csv"
        TargetName = "camelyonpatch_level_2_split_test_meta.csv"
        Md5 = "67589e00a4a37ec317f2d1932c7502ca"
        Compressed = $false
    }
)

function Invoke-ResumableDownload {
    param(
        [Parameter(Mandatory)] [string]$SourceName,
        [Parameter(Mandatory)] [string]$OutputPath,
        [Parameter(Mandatory)] [string]$ExpectedMd5
    )

    $url = "https://zenodo.org/records/2546921/files/$SourceName`?download=1"
    $partialPath = "$OutputPath.part"

    Write-Host "Downloading $SourceName"
    & curl.exe `
        --location `
        --fail `
        --retry 8 `
        --retry-delay 10 `
        --continue-at - `
        --output $partialPath `
        $url

    if ($LASTEXITCODE -ne 0) {
        throw "curl failed for $SourceName with exit code $LASTEXITCODE"
    }

    $actualMd5 = (Get-FileHash -LiteralPath $partialPath -Algorithm MD5).Hash.ToLowerInvariant()
    if ($actualMd5 -ne $ExpectedMd5) {
        throw "MD5 mismatch for $SourceName. Expected $ExpectedMd5, obtained $actualMd5."
    }

    Move-Item -LiteralPath $partialPath -Destination $OutputPath -Force
    Write-Host "MD5 verified: $ExpectedMd5"
}

function Expand-GzipFile {
    param(
        [Parameter(Mandatory)] [string]$ArchivePath,
        [Parameter(Mandatory)] [string]$OutputPath
    )

    $partialOutput = "$OutputPath.partial"
    if (Test-Path -LiteralPath $partialOutput) {
        Remove-Item -LiteralPath $partialOutput -Force
    }

    Write-Host "Extracting $(Split-Path $ArchivePath -Leaf) -> $(Split-Path $OutputPath -Leaf)"
    $inputStream = [System.IO.File]::OpenRead($ArchivePath)
    try {
        $gzipStream = [System.IO.Compression.GZipStream]::new(
            $inputStream,
            [System.IO.Compression.CompressionMode]::Decompress
        )
        try {
            $outputStream = [System.IO.File]::Create($partialOutput)
            try {
                $gzipStream.CopyTo($outputStream, 4MB)
            }
            finally {
                $outputStream.Dispose()
            }
        }
        finally {
            $gzipStream.Dispose()
        }
    }
    finally {
        $inputStream.Dispose()
    }

    Move-Item -LiteralPath $partialOutput -Destination $OutputPath -Force
}

foreach ($file in $files) {
    $targetPath = Join-Path $Destination $file.TargetName
    if (Test-Path -LiteralPath $targetPath) {
        Write-Host "Already present, skipping: $($file.TargetName)"
        continue
    }

    if ($file.Compressed) {
        $archivePath = Join-Path $Destination $file.SourceName
        if (-not (Test-Path -LiteralPath $archivePath)) {
            Invoke-ResumableDownload `
                -SourceName $file.SourceName `
                -OutputPath $archivePath `
                -ExpectedMd5 $file.Md5
        }
        else {
            $actualMd5 = (Get-FileHash -LiteralPath $archivePath -Algorithm MD5).Hash.ToLowerInvariant()
            if ($actualMd5 -ne $file.Md5) {
                throw "Existing archive has invalid MD5: $archivePath"
            }
        }

        Expand-GzipFile -ArchivePath $archivePath -OutputPath $targetPath
        if (-not $KeepArchives) {
            Remove-Item -LiteralPath $archivePath -Force
        }
    }
    else {
        Invoke-ResumableDownload `
            -SourceName $file.SourceName `
            -OutputPath $targetPath `
            -ExpectedMd5 $file.Md5
    }
}

Write-Host "PCam download and extraction completed: $Destination"
