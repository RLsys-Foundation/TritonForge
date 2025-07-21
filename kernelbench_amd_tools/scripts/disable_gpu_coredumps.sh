#!/bin/bash
# Script to disable GPU core dumps for KernelBench evaluation

echo "Disabling GPU core dumps to prevent large files..."

# Disable core dumps for the current session
ulimit -c 0

# Disable AMD GPU core dumps specifically
export HSA_ENABLE_COREDUMP=0
export AMD_LOG_LEVEL=0

# Disable ROCm crash dumps
export ROCM_DISABLE_CRASH_DUMP=1

# Also set for HIP
export HIP_ENABLE_COREDUMP=0

echo "GPU core dumps disabled. Environment variables set:"
echo "  HSA_ENABLE_COREDUMP=0"
echo "  AMD_LOG_LEVEL=0"
echo "  ROCM_DISABLE_CRASH_DUMP=1"
echo "  HIP_ENABLE_COREDUMP=0"
echo "  ulimit -c = $(ulimit -c)"