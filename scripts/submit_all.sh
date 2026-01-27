#!/bin/bash

cd /home/bates.car/context_har

filepath='job_scripts/experiments_deepconvcontext_bilstm'

echo "Running $filepath"

counter=1
mapfile -t lines < <(grep -v '^$' "$filepath")

for ((i=0; i<${#lines[@]}; i+=3)); do
    if [ $((i+2)) -lt ${#lines[@]} ]; then
        cmd1="${lines[i]}"
        cmd2="${lines[i+1]}"
        cmd3="${lines[i+2]}"
        
        echo "Submitting job $counter with 3 commands:"
        echo "  1: $cmd1"
        echo "  2: $cmd2"
        echo "  3: $cmd3"
        
        sbatch --job-name="bilstm_$counter" job_template.sh "$cmd1" "$cmd2" "$cmd3"
        ((counter++))
    fi
done

echo "Submitted $((counter-1)) jobs total!"