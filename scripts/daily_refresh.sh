#!/usr/bin/env bash
# daily_refresh.sh — Pull fresh current-season data and rebuild predictions.
#
# Usage:
#   ./scripts/daily_refresh.sh              # daily refresh (current season only)
#   ./scripts/daily_refresh.sh full         # full rebuild (all seasons from scratch)
#
# Scheduling (macOS launchd):
#   cp scripts/com.68bracket.daily-refresh.plist ~/Library/LaunchAgents/
#   launchctl load ~/Library/LaunchAgents/com.68bracket.daily-refresh.plist
#
# To unload:
#   launchctl unload ~/Library/LaunchAgents/com.68bracket.daily-refresh.plist

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PROJECT_DIR/.venv/bin/python"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/refresh_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "=== 68bracket daily refresh ==="
log "Project: $PROJECT_DIR"

MODE="${1:-daily}"

cd "$PROJECT_DIR"

if [[ "$MODE" == "daily" ]]; then
    log "Running: scrape --current-only (current season only, force-refresh)"
    "$VENV" -m main scrape --current-only >> "$LOG_FILE" 2>&1

    log "Running: build features"
    "$VENV" -m main build >> "$LOG_FILE" 2>&1

    log "Running: train models"
    "$VENV" -m main train --model ensemble >> "$LOG_FILE" 2>&1

    log "Running: predict"
    "$VENV" -m main predict --model ensemble >> "$LOG_FILE" 2>&1

    log "Daily refresh complete."
elif [[ "$MODE" == "full" ]]; then
    log "Running: full pipeline (all seasons from scratch)"
    "$VENV" -m main all --model ensemble >> "$LOG_FILE" 2>&1
    log "Full pipeline complete."
else
    log "Unknown mode: $MODE (use 'daily' or 'full')"
    exit 1
fi

# Clean up logs older than 30 days
find "$LOG_DIR" -name "refresh_*.log" -mtime +30 -delete 2>/dev/null || true

log "=== Done ==="
