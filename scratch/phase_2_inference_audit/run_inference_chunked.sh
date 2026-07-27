#!/bin/bash
# Wrapper to run VoxTell Inference in chunks and bypass memory leaks

TOTAL_VOLUMES=200
CHUNK_SIZE=10
SPLIT="val"
TILE_STEP=0.5
PYTHON_BIN="./.venv-voxtell/bin/python"
SCRIPT="scripts/voxtell/voxtell_inference.py"

echo "==========================================="
echo "Starting Chunked Inference for $SPLIT split"
echo "Total Volumes: $TOTAL_VOLUMES | Chunk Size: $CHUNK_SIZE"
echo "==========================================="

for (( i=0; i<TOTAL_VOLUMES; i+=CHUNK_SIZE )); do
    end_idx=$((i + CHUNK_SIZE))
    if [ $end_idx -gt $TOTAL_VOLUMES ]; then
        end_idx=$TOTAL_VOLUMES
    fi
    
    echo ""
    echo ">>> Running chunk [$i : $end_idx]"
    
    # Run python script for the specific chunk
    $PYTHON_BIN $SCRIPT --split $SPLIT --tile_step_size $TILE_STEP --start_idx $i --end_idx $end_idx
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "!!! Chunk [$i : $end_idx] failed with exit code $EXIT_CODE. Stopping."
        exit $EXIT_CODE
    fi
    
    echo ">>> Chunk [$i : $end_idx] completed successfully. OS Memory has been fully reclaimed."
done

echo "==========================================="
echo "All chunks completed successfully!"
echo "==========================================="
