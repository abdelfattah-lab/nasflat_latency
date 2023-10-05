#!/bin/bash

collect_info() {
    # Get CPU info
    cpu_name=$(lscpu | grep 'Model name:' | awk -F: '{print $2}' | xargs)
    num_cores=$(lscpu | grep '^CPU(s):' | awk '{print $2}')
    num_threads=$(lscpu | grep 'Thread(s) per core:' | awk '{print $4}' | xargs)
    cpu_cache=$(lscpu | grep 'L3 cache:' | awk '{print $3}' | xargs)
    
    # Get RAM info
    total_ram=$(free -h | grep 'Mem:' | awk '{print $2}')
    ram_freq=$(dmidecode -t memory | grep 'Speed:' | head -1 | awk '{print $2}' | xargs)
    
    # Get Disk I/O speeds (This is a simple benchmark, consider using a more comprehensive tool like fio)
    disk_io=$(dd if=/dev/zero of=tempfile bs=1M count=1024 conv=fdatasync,notrunc 2>&1 | grep 'bytes' | awk '{print $8 " " $9}')
    rm -f tempfile
    
    # Get Network speeds (Consider using a more appropriate tool or method depending on your network setup)
    network_speed=$(cat /sys/class/net/*/speed 2> /dev/null | xargs)
    
    # Get Operating System Details
    os_details=$(uname -a)
    
    # Get GPU info
    gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | xargs)
    num_gpus=$(nvidia-smi --list-gpus | wc -l)
    
    echo "$1,$cpu_name,$num_cores,$num_threads,$total_ram,$ram_freq,$cpu_cache,$disk_io,$network_speed,$os_details,$gpu_info,$num_gpus"
}

collect_info $1
