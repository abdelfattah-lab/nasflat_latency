
#!/bin/bash

# Create/overwrite the file
echo "" > multipredict_unified_joblist.log

# Loop to generate the required lines
for start in $(seq 0 50 4950); do
    end=$((start + 50))
    # Note: other values in the log entry (like "True,helptest,True,30000,4") are assumed to be constants based on your example
    echo "True,helptest,True,30000,4,python fbnet_model_initialzier.py --start $start --end $end" >> multipredict_unified_joblist.log
done

