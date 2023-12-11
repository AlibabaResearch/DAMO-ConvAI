#!/bin/bash

function get_gpu_id() {
    gpu_node=$1
    selected_gpus=""
    gpu_array=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15")
    for(( i=0;i<${gpu_node};i++ )) do
        if [[ ${selected_gpus} == "" ]]; then
            selected_gpus=${gpu_array[i]}
        else
            selected_gpus=${selected_gpus}","${gpu_array[i]}
        fi
    done;
    echo "${selected_gpus}"
}
