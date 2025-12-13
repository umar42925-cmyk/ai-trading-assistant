# === Personal AI Backup Script ===

$ProjectDir = "C:\Users\LCH\personal-ai"
$BackupRoot = "C:\Users\LCH\AI_Backups"

# Create backup folder if it doesn't exist
if (!(Test-Path $BackupRoot)) {
    New-Item -ItemType Directory -Path $BackupRoot
}

# Timestamp
$Timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$BackupDir = "$BackupRoot\personal-ai_$Timestamp"

# Copy essential files
New-Item -ItemType Directory -Path $BackupDir
Copy-Item "$ProjectDir\main.py" $BackupDir
Copy-Item "$ProjectDir\memory.json" $BackupDir
Copy-Item "$ProjectDir\pyproject.toml" $BackupDir

Write-Output "Backup completed: $BackupDir"
