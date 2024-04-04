#!/usr/bin/env bash


if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    
    echo "Process script for processing bulk data at once!"
    if [[ $(which python) ]]; then
        python ./src/scan.py -h
    else
        python3 ./src/scan.py -h
    fi
    
    echo "--dir target Directory with data"
    
    exit
    
fi

# Initialize variable to hold the directory path
DIR=""

# Loop through the command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dir) # Look for the --dir option
            DIR="$2" # Assign the next argument as the directory path
            shift # Move past the value
        ;;
        --gui) # Handle any other arguments
            echo "its not recommmanded to use interactive mode when process bulk data"
            exit 1
        ;;
    esac
    shift # Move to the next argument
done


echo "processing $1... non interactive"

for image in $DIR/*; do
    
    echo $image
    if [[ $(which python) ]]; then
        python ./src/scan.py --image "./$image" "$@"
    else
        python ./src/scan.py --image "./$image" "$@"
    fi
    
done




