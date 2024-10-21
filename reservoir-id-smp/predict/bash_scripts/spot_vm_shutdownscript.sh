PID=$(tsp -p)
kill -2 "$PID"
touch '~/shutdown_run.foo'
