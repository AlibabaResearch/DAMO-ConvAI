#!/bin/bash
set +x

source "$(dirname "$0")/../scripts/env_setup.sh"
pre_setup_environment

pip install -r requirements_common.txt --ignore-requires-python

run_file=$1
chmod +x "$run_file"

sh $run_file
exit_status=$?

if [ $exit_status -ne 0 ]; then
    echo "执行脚本失败，退出状态: $exit_status" >&2
    exit 137
fi
exit 0

