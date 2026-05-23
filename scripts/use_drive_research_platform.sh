#!/usr/bin/env bash
# Source this file when you want local terminals/apps to use the Drive copy.
#
#   source scripts/use_drive_research_platform.sh
#
export RESEARCH_PLATFORM_STORAGE_MODE="drive"
export RESEARCH_PLATFORM_ROOT="/Users/itsgennymac/Library/CloudStorage/GoogleDrive-sfn.gns@gmail.com/Il mio Drive/GitHub/machine-learning-for-trading/research_platform_definitive"
export FINANCIAL_DB_ROOT="/Users/itsgennymac/Library/CloudStorage/GoogleDrive-sfn.gns@gmail.com/Il mio Drive/Database Finanziario"
export DB_BASE="$FINANCIAL_DB_ROOT"
export DATA_PATH="$FINANCIAL_DB_ROOT"
export RESEARCH_PLATFORM_OUTPUT_ROOT="$RESEARCH_PLATFORM_ROOT/output"
export RESEARCH_PLATFORM_LOCAL_CACHE="$RESEARCH_PLATFORM_ROOT/output/data_cache"
echo "Drive-first research platform enabled"
echo "RESEARCH_PLATFORM_ROOT=$RESEARCH_PLATFORM_ROOT"
echo "FINANCIAL_DB_ROOT=$FINANCIAL_DB_ROOT"
