#!/bin/sh
exec uvicorn predict:predict_cpu_tf_rf --host 0.0.0.0 --port "${PORT}"