$ErrorActionPreference = 'Stop'
$taskOs = Get-CimInstance Win32_OperatingSystem
$taskCpu = Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors
$taskVideo = Get-CimInstance Win32_VideoController | Select-Object Name
$taskReport = [ordered]@{
    observed_at = (Get-Date).ToUniversalTime().ToString('o')
    os = $taskOs.Caption
    shell = $PSVersionTable.PSVersion.ToString()
    ram_total_gib = [math]::Round($taskOs.TotalVisibleMemorySize / 1MB, 2)
    ram_available_gib = [math]::Round($taskOs.FreePhysicalMemory / 1MB, 2)
    cpu = $taskCpu
    video = $taskVideo
    nvidia_smi_available = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
    github_cli_available = [bool](Get-Command gh -ErrorAction SilentlyContinue)
}
$taskReport | ConvertTo-Json -Depth 5
if ($taskReport.nvidia_smi_available) {
    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
}
py -0p
git --version
git status --short --branch
git remote -v
