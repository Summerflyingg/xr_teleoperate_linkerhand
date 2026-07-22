#!/usr/bin/env bash
set -Eeuo pipefail

ETH_IFACE="enp87s0"
SERVER_ADDR="192.168.123.2/24"
OLD_ADDR="192.168.1.99/24"
ROBOT_ARM_ADDR="192.168.123.161"
CAN_INTERFACES=(can0 can1)

echo "正在申请 sudo 权限……"
sudo -v

echo "[1/4] 配置机器人手臂直连网口 $ETH_IFACE……"
if ! ip link show "$ETH_IFACE" >/dev/null 2>&1; then
    echo "错误：找不到有线网卡 $ETH_IFACE。" >&2
    exit 1
fi

sudo ip link set "$ETH_IFACE" up
sudo ip addr del "$OLD_ADDR" dev "$ETH_IFACE" 2>/dev/null || true
sudo ip addr replace "$SERVER_ADDR" dev "$ETH_IFACE"

ip -br addr show "$ETH_IFACE"
ip route get "$ROBOT_ARM_ADDR"

echo "[2/4] 检查机器人手臂控制网络……"
if ! ping -I "$ETH_IFACE" -c 3 -W 1 "$ROBOT_ARM_ADDR"; then
    echo "错误：无法通过 $ETH_IFACE 访问 $ROBOT_ARM_ADDR。" >&2
    echo "请检查机器人颈部网线、机器人电源以及网口指示灯。" >&2
    exit 1
fi

echo "[3/4] 加载 PEAK 驱动并初始化灵巧手 CAN……"
sudo modprobe peak_usb
sudo udevadm settle

for _ in $(seq 1 5); do
    [[ -e /sys/class/net/can0 && -e /sys/class/net/can1 ]] && break
    sleep 1
done

for can_iface in "${CAN_INTERFACES[@]}"; do
    if ! ip link show "$can_iface" >/dev/null 2>&1; then
        echo "错误：找不到 $can_iface。当前检测到的 PEAK USB 设备：" >&2
        lsusb | grep -Ei 'PEAK|0c72' >&2 || true
        exit 1
    fi

    sudo ip link set "$can_iface" down
    sudo ip link set "$can_iface" up type can bitrate 1000000 restart-ms 100
done

echo "[4/4] 检查 CAN 状态……"
for can_iface in "${CAN_INTERFACES[@]}"; do
    ip -br link show "$can_iface"
    ip -details link show "$can_iface" \
        | grep -E 'can state|bitrate ' \
        | sed 's/^[[:space:]]*/  /'

    usb_path="$(udevadm info -q property -p "/sys/class/net/$can_iface" \
        | sed -n 's/^ID_PATH=//p' \
        | head -n 1)"
    echo "  USB path: ${usb_path:-unknown}"
done

echo
echo "服务器端网络与 CAN 初始化完成。"
echo "手臂网络：$ETH_IFACE -> $ROBOT_ARM_ADDR"
echo "灵巧手 CAN：can0、can1，1 Mbps，restart-ms=100"
echo
echo "重要：遥操程序当前默认左手=can0、右手=can1。"
echo "两块 PCAN-USB 没有唯一序列号，重插或换 USB 口后编号可能交换。"
echo "首次动作时请低速、逐只手确认；若左右相反，请立即停止遥操。"
