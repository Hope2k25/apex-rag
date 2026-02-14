# Powershell script to start Neo4j Community Edition with the correct Java Environment
# Usage: .\scripts\start_neo4j.ps1

Write-Host "Setting up Java Environment..." -ForegroundColor Cyan

# Set JAVA_HOME to the Microsoft OpenJDK we installed
$env:JAVA_HOME = "C:\Program Files\Microsoft\jdk-25.0.2.10-hotspot"
$env:Path = "$env:JAVA_HOME\bin;" + $env:Path

# Verify Java
java -version

Write-Host "Starting Neo4j Console..." -ForegroundColor Green
Write-Host "Keep this window OPEN to keep the database running." -ForegroundColor Yellow

# Path to Neo4j
$neo4jPath = "C:\Users\skrae\Desktop\Dev Tools\rag_and_memory\neo4j-community-2025.12.1"

if (Test-Path $neo4jPath) {
    Set-Location $neo4jPath
    .\bin\neo4j.bat console
} else {
    Write-Error "Neo4j directory not found at $neo4jPath"
}
