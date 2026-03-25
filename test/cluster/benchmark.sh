#!/usr/bin/env bash
set -euo pipefail

TEST_PATH="${1:-test/cluster/test_lwt_benchmark_v2.py}"

OLD_GOV=""
OLD_TURBO=""

if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
    OLD_GOV="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
fi

if [[ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]]; then
    OLD_TURBO="$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)"
fi

cleanup() {
    echo "[perf] restoring CPU settings"

    if [[ -n "${OLD_GOV}" ]]; then
        sudo cpupower frequency-set --governor "${OLD_GOV}" >/dev/null || true
    fi

    if [[ -n "${OLD_TURBO}" ]] && [[ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]]; then
        echo "${OLD_TURBO}" | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo >/dev/null || true
    fi
}

trap cleanup EXIT

echo "[perf] setting governor=performance"
sudo cpupower frequency-set --governor performance

if [[ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]]; then
    echo "[perf] disabling turbo"
    echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo >/dev/null
fi

./tools/toolchain/dbuild pytest --test-py-init --mode=dev "$TEST_PATH" --repeat=30