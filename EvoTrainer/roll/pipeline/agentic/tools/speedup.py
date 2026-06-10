# speedup.py

# 统一的 APT 加速配置脚本模板
setup_apt_source_template = """
#!/bin/bash

detect_system_and_version() {{
    if [ -f /etc/debian_version ]; then
        . /etc/os-release
        if [ "$ID" = "ubuntu" ]; then
            echo "ubuntu:$VERSION_CODENAME"
        elif [ "$ID" = "debian" ]; then
            echo "debian:$VERSION_CODENAME"
        else
            echo "unknown:"
        fi
    else
        echo "unknown:"
    fi
}}

setup_apt_source() {{
    SYSTEM_INFO=$(detect_system_and_version)
    SYSTEM=$(echo "$SYSTEM_INFO" | cut -d: -f1)
    CODENAME=$(echo "$SYSTEM_INFO" | cut -d: -f2)
    echo "系统类型: $SYSTEM, 版本代号: $CODENAME"

    # 备份原始源文件
    if [ ! -f /etc/apt/sources.list.backup ]; then
        cp /etc/apt/sources.list /etc/apt/sources.list.backup
    fi

    if [ "$SYSTEM" = "debian" ]; then
        if [ -z "$CODENAME" ]; then
            CODENAME="bookworm"
        fi
        cat > /etc/apt/sources.list <<EOF
deb http://{mirror_base}/debian/ ${{CODENAME}} main non-free non-free-firmware contrib
deb http://{mirror_base}/debian-security/ ${{CODENAME}}-security main
deb http://{mirror_base}/debian/ ${{CODENAME}}-updates main non-free non-free-firmware contrib
EOF
    elif [ "$SYSTEM" = "ubuntu" ]; then
        if [ -z "$CODENAME" ]; then
            if [ -f /etc/os-release ]; then
                VERSION_ID=$(grep VERSION_ID /etc/os-release | cut -d'"' -f2)
                case "$VERSION_ID" in
                    "24.04") CODENAME="noble" ;;
                    "22.04") CODENAME="jammy" ;;
                    "20.04") CODENAME="focal" ;;
                    *) CODENAME="noble" ;;
                esac
            else
                CODENAME="noble"
            fi
        fi
        cat > /etc/apt/sources.list <<EOF
deb http://{mirror_base}/ubuntu/ $CODENAME main restricted universe multiverse
deb http://{mirror_base}/ubuntu/ $CODENAME-security main restricted universe multiverse
deb http://{mirror_base}/ubuntu/ $CODENAME-updates main restricted universe multiverse
deb http://{mirror_base}/ubuntu/ $CODENAME-backports main restricted universe multiverse
EOF
    fi

    # 清理可能存在的其他源文件
    rm -rf /etc/apt/sources.list.d

    # 设置APT配置，加速下载
    cat > /etc/apt/apt.conf.d/99speedup <<EOF
Acquire::http::Timeout "30";
Acquire::ftp::Timeout "30";
Acquire::Retries "3";
APT::Acquire::Retries "3";
APT::Get::Assume-Yes "true";
APT::Install-Recommends "false";
APT::Install-Suggests "false";
EOF

    # 清理APT缓存并更新
    apt-get clean
    rm -rf /var/lib/apt/lists/*
    echo ">>> APT源配置完成"
}}

setup_apt_source
apt-get update
"""

# APT 加速配置脚本（阿里云公网源）
setup_public_apt_source = setup_apt_source_template.format(mirror_base="mirrors.example.com")

# APT 加速配置脚本（阿里云内网源）
setup_internal_apt_source = setup_apt_source_template.format(mirror_base="mirrors.example.com")

# PIP 加速配置脚本（阿里云源）
setup_pip_source = """
#!/bin/bash

setup_pip_source() {
    echo ">>> 配置阿里云pip源..."

    # 为 root 用户配置
    mkdir -p /root/.pip
    cat > /root/.pip/pip.conf <<EOF
[global]
index-url = http://mirrors.example.com/pypi/simple/
trusted-host = mirrors.example.com
timeout = 120

[install]
trusted-host = mirrors.example.com
EOF

    # 为其他可能存在的用户配置
    for home_dir in /home/*; do
        if [ -d "$home_dir" ]; then
            username=$(basename "$home_dir")
            mkdir -p "$home_dir/.pip"
            cat > "$home_dir/.pip/pip.conf" <<EOF
[global]
index-url = http://mirrors.example.com/pypi/simple/
trusted-host = mirrors.example.com
timeout = 120

[install]
trusted-host = mirrors.example.com
EOF
            chown -R "$username:$username" "$home_dir/.pip" 2>/dev/null || true
        fi
    done

    echo ">>> pip源配置完成"
}

setup_pip_source
"""
