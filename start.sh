#!/bin/bash
PORT=${PORT:-5001}
exec gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
