#!/bin/bash

pre_setup_environment() {

    MASTER_ADDR=$MASTER_ADDR
    MASTER_PORT=$MASTER_PORT
    RANK=$RANK
    WORLD_SIZE=$WORLD_SIZE

    unset LD_PRELOAD
    export NCCL_IB_DISABLE=0
    export NCCL_IB_HCA=mlx5
    export NCCL_IB_GID_INDEX=3
    if [[ "${WORLD_SIZE:-1}" -gt 1 ]]; then
        export NCCL_SOCKET_IFNAME=eth0
        export GLOO_SOCKET_IFNAME=eth0
    else
        export NCCL_SOCKET_IFNAME=bond0
        export GLOO_SOCKET_IFNAME=bond0
    fi
    export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 # system may overwrite JAVA_HOME

    echo "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
    echo "GLOO_SOCKET_IFNAME: $GLOO_SOCKET_IFNAME"


    # 重新挂载/dev/shm以增加大小和inode数量
    echo "Remounting /dev/shm with increased size..."
    mount -o size=81920M -o nr_inodes=1000000 -o noatime,nodiratime -o remount /dev/shm

    echo "init env: "
    echo "MASTER_ADDR: ${MASTER_ADDR}"
    echo "MASTER_PORT: ${MASTER_PORT}"
    echo "RANK: ${RANK}"
    echo "WORLD_SIZE: ${WORLD_SIZE}"

    mkdir -p /data/oss_bucket_0/mdl_schduler_mock
    scheduler_file=/data/oss_bucket_0/mdl_schduler_mock/${MASTER_ADDR}.txt

    # 主节点(RANK=0)创建调度文件
    if [[ "$RANK" -eq 0 ]]; then
        echo "Installing iproute2..."
        apt update
        apt install -y iproute2
        echo "Resolving MASTER_ADDR from $MASTER_ADDR..."
        MASTER_ADDR=$(ip -4 addr show bond0 | grep -oP 'inet \K[\d.]+')
        if [ -z "$MASTER_ADDR" ]; then
            MASTER_ADDR=$(ip -4 addr show eth0 | grep -oP 'inet \K[\d.]+')
        fi
        echo "parsed MASTER_ADDR: ${MASTER_ADDR}"

        echo "Creating scheduler file: ${scheduler_file}"
        echo "$MASTER_ADDR" > "$scheduler_file"
        echo "$MASTER_PORT" >> "$scheduler_file"
        echo "Scheduler file ${scheduler_file} created with MASTER_ADDR=${MASTER_ADDR} and MASTER_PORT=${MASTER_PORT}"
    else
        # 工作节点等待调度文件
        echo "Waiting for scheduler file: ${scheduler_file} ..."
        wait_attempts=0
        wait_interval=10  # 秒
        while [[ ! -f "$scheduler_file" ]]; do
            sleep "$wait_interval"
            ((wait_attempts++))
            echo "Waiting... ($wait_attempts) ${wait_interval}秒"
        done

        # 从文件读取MASTER_ADDR和PORT
        echo "Scheduler file found, reading configuration..."
        read -r MASTER_ADDR < "$scheduler_file"
        read -r MASTER_PORT < <(sed -n '2p' "$scheduler_file")

        echo "Updated from scheduler file:"
        echo "MASTER_ADDR: ${MASTER_ADDR}"
        echo "MASTER_PORT: ${MASTER_PORT}"
    fi

    export MASTER_ADDR
    export MASTER_PORT

}

