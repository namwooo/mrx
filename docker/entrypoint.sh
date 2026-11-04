#!/usr/bin/env bash
# Runs the API server and one worker per job type together in one container.
#
# - The API starts first, because it creates the DB tables on startup.
# - The API must stay reachable: if a worker exits (crash or not), only that worker is restarted,
#   after WORKER_RESTART_DELAY seconds.
# - If the API exits, the workers are stopped and the container exits, so the restart policy
#   (docker run --restart unless-stopped) brings the container back.
# - SIGTERM (docker stop) is passed to all. A worker finishes its current job before exiting.
#   A job still running after WORKER_STOP_GRACE seconds is stopped and recorded as failed, before
#   docker kills the container (docker run --stop-timeout must be longer than WORKER_STOP_GRACE).
set -uo pipefail

PORT="${PORT:-8000}"
WORKERS="${WORKERS:-init_training labeling model_export training}"
WORKER_RESTART_DELAY="${WORKER_RESTART_DELAY:-5}"
WORKER_STOP_GRACE="${WORKER_STOP_GRACE:-100}"

requested_stop=0  # set by docker stop / Ctrl+C
stopped=0
api_pid=""
grace_pid=""
declare -A worker_pids=()  # job type -> pid

worker_pid_list() {
    echo "${worker_pids[@]+"${worker_pids[@]}"}"
}

stop_all() {
    # Send SIGTERM only once, even if called again.
    [ "$stopped" = 1 ] && return
    stopped=1
    # shellcheck disable=SC2046
    kill -TERM ${api_pid:+"$api_pid"} $(worker_pid_list) 2>/dev/null
    (
        sleep "$WORKER_STOP_GRACE"
        # shellcheck disable=SC2046
        kill -TERM $(worker_pid_list) 2>/dev/null
    ) &
    grace_pid=$!
}

on_signal() {
    requested_stop=1
    stop_all
}
trap on_signal TERM INT

start_worker() {
    python -m workers "$1" &
    worker_pids["$1"]=$!
}

uvicorn app.main:app --host 0.0.0.0 --port "$PORT" &
api_pid=$!

# Wait until the API answers (so the tables exist) before starting the workers.
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

for job_type in $WORKERS; do
    start_worker "$job_type"
done

status=0
while true; do
    # Returns when any process exits, or when a signal arrives.
    # shellcheck disable=SC2046
    wait -n "$api_pid" $(worker_pid_list)
    status=$?

    if [ "$requested_stop" = 1 ]; then
        break
    fi

    if ! kill -0 "$api_pid" 2>/dev/null; then
        echo "entrypoint: API server exited (status $status), stopping the container" >&2
        break
    fi

    exited=()
    for job_type in "${!worker_pids[@]}"; do
        if ! kill -0 "${worker_pids[$job_type]}" 2>/dev/null; then
            exited+=("$job_type")
            unset "worker_pids[$job_type]"  # it is gone; do not signal a pid the OS may hand to another process
        fi
    done

    if [ "${#exited[@]}" -gt 0 ]; then
        echo "entrypoint: ${exited[*]} worker exited (status $status), restarting in ${WORKER_RESTART_DELAY}s" >&2
        # sleep in the background + wait, so that docker stop is handled right away
        sleep "$WORKER_RESTART_DELAY" &
        wait $!
        if [ "$requested_stop" = 1 ]; then
            break
        fi
        for job_type in "${exited[@]}"; do
            start_worker "$job_type"
        done
    fi
done

stop_all
# shellcheck disable=SC2046
wait "$api_pid" $(worker_pid_list)
[ -n "$grace_pid" ] && kill "$grace_pid" 2>/dev/null
if [ "$requested_stop" = 1 ]; then
    status=0  # stopped on request, not a failure
fi
echo "entrypoint: stopped (status $status)" >&2
exit "$status"
