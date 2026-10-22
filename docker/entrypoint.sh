#!/usr/bin/env bash
# Runs the API server and the batch worker together in one container.
#
# - The API starts first, because it creates the DB tables on startup.
# - The API must stay reachable: if the worker exits (crash or not), only the worker is restarted,
#   after WORKER_RESTART_DELAY seconds.
# - If the API exits, the worker is stopped and the container exits, so the restart policy
#   (docker run --restart unless-stopped) brings the container back.
# - SIGTERM (docker stop) is passed to both. The worker finishes its current job before exiting.
set -uo pipefail
 
PORT="${PORT:-8000}"
WORKER_RESTART_DELAY="${WORKER_RESTART_DELAY:-5}"
 
requested_stop=0  # set by docker stop / Ctrl+C
stopped=0
api_pid=""
worker_pid=""
 
stop_all() {
    # Send SIGTERM only once, even if called again.
    [ "$stopped" = 1 ] && return
    stopped=1
    kill -TERM ${api_pid:+"$api_pid"} ${worker_pid:+"$worker_pid"} 2>/dev/null
}
 
on_signal() {
    requested_stop=1
    stop_all
}
trap on_signal TERM INT
 
start_worker() {
    python -m workers batch &
    worker_pid=$!
}
 
uvicorn app.main:app --host 0.0.0.0 --port "$PORT" &
api_pid=$!
 
# Wait until the API answers (so the tables exist) before starting the worker.
for _ in $(seq 1 60); do
    if python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/openapi.json')" 2>/dev/null; then
        break
    fi
    if ! kill -0 "$api_pid" 2>/dev/null || [ "$requested_stop" = 1 ]; then
        echo "entrypoint: API server exited during startup" >&2
        exit 1
    fi
    sleep 1
done
 
start_worker
 
status=0
while true; do
    # Returns when either process exits, or when a signal arrives.
    wait -n "$api_pid" "$worker_pid"
    status=$?
 
    if [ "$requested_stop" = 1 ]; then
        break
    fi
 
    if ! kill -0 "$api_pid" 2>/dev/null; then
        echo "entrypoint: API server exited (status $status), stopping the container" >&2
        break
    fi
 
    if ! kill -0 "$worker_pid" 2>/dev/null; then
        echo "entrypoint: worker exited (status $status), restarting it in ${WORKER_RESTART_DELAY}s" >&2
        worker_pid=""  # it is gone; do not signal a pid the OS may hand to another process
        # sleep in the background + wait, so that docker stop is handled right away
        sleep "$WORKER_RESTART_DELAY" &
        wait $!
        if [ "$requested_stop" = 1 ]; then
            break
        fi
        start_worker
    fi
done
 
stop_all
wait
if [ "$requested_stop" = 1 ]; then
    status=0  # stopped on request, not a failure
fi
echo "entrypoint: stopped (status $status)" >&2
exit "$status"