#!/bin/bash
# Start/stop the board-side Ethernet-to-BRAM keyword bridge.

set -euo pipefail
cd "$(dirname "$0")"

PY=/usr/local/share/pynq-venv/bin/python3
PIDFILE="$PWD/keyword_bridge.pid"
LOGFILE="$PWD/keyword_bridge.log"

stop_service() {
    if [ -f "$PIDFILE" ]; then
        pid="$(cat "$PIDFILE")"
        if sudo -n kill -0 "$pid" 2>/dev/null; then
            sudo -n kill "$pid"
            for _ in 1 2 3 4 5; do
                sudo -n kill -0 "$pid" 2>/dev/null || break
                sleep 0.2
            done
        fi
        sudo -n rm -f "$PIDFILE"
    fi
}

case "${1:-status}" in
    start)
        stop_service
        sudo -n bash -c "source /etc/profile.d/xrt_setup.sh; cd '$PWD'; \
            nohup '$PY' keyword_bridge.py --bind 'tcp://*:5556' \
            < /dev/null > '$LOGFILE' 2>&1 & echo \$! > '$PIDFILE'"
        sleep 1
        pid="$(cat "$PIDFILE")"
        sudo -n kill -0 "$pid"
        echo "keyword bridge started (pid $pid)"
        cat "$LOGFILE"
        ;;
    stop)
        stop_service
        echo "keyword bridge stopped"
        ;;
    status)
        if [ -f "$PIDFILE" ] && sudo -n kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            echo "keyword bridge running (pid $(cat "$PIDFILE"))"
            cat "$LOGFILE"
        else
            echo "keyword bridge not running"
            exit 1
        fi
        ;;
    log)
        cat "$LOGFILE"
        ;;
    *)
        echo "usage: $0 {start|stop|status|log}" >&2
        exit 2
        ;;
esac
