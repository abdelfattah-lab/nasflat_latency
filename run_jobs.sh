#!/bin/bash

# Get the node list automatically from sinfo
nodes=$(sinfo -h -o %N | tr ',' '\n' | xargs -n 1 scontrol show hostname)

logfile="g2_server_info.log"

# Header for logfile
echo "NodeName,CPUName,NumCores,NumThreads,TotalRAM,RAMFrequency,CPUCache,DiskIOSpeed,NetworkSpeed,OSDetails,GPUModel,GPUVRAM,NumGPUs" > $logfile

# Loop over nodes and launch jobs
for node in $nodes
do
    # Launch job to collect information on the node and append to logfile
    sbatch --requeue --nodelist=$node --gres=gpu:1 --wrap="bash collect_info.sh $node >> $logfile"
done
