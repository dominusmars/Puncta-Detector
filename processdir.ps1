# PowerShell equivalent of the Bash script


if ($args[0] -eq "-h" -or $args[0] -eq "--help") {
    Write-Host "Process script for processing bulk data at once!"
    if (Get-Command "python" -ErrorAction SilentlyContinue) {
        python ./src/scan.py -h
    } else {
        python3 ./src/scan.py -h
    }
    Write-Host "--dir target Directory with data"
    exit
}

# Initialize variable to hold the directory path
$DIR = ""

# Loop through the command-line arguments
for ($i = 0; $i -lt $args.Length; $i++) {
    switch ($args[$i]) {
        "--dir" {
            $DIR = $args[$i+1] # Assign the next argument as the directory path
            $i++ # Skip the next argument since it's the value for --dir
        }
        "--gui" {
            Write-Host "It's not recommended to use interactive mode when processing bulk data."
            exit 1
        }
        default {
            Write-Host "Unknown option: $($args[$i])"
            exit 1
        }
    }
}

if (-not $DIR) {
    Write-Host "No directory specified. Use --dir to specify the target directory."
    exit 1
}

Write-Host "Processing non-interactive..."

# Process each image in the directory
Get-ChildItem -Path "$dir\*" | ForEach-Object {
    $image = $_.FullName
    Write-Host "Processing $image..."
    if (Get-Command "python" -ErrorAction SilentlyContinue) {
        python ./src/scan.py --image "$image" $arguments
    } else {
        python3 ./src/scan.py --image "$image" $arguments
    }
}
