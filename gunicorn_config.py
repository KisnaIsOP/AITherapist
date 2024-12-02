import multiprocessing
import os

# Worker configuration
bind = "0.0.0.0:" + str(os.getenv("PORT", "8000"))
worker_class = "aiohttp.worker.GunicornWebWorker"
workers = multiprocessing.cpu_count() * 2 + 1
threads = 2
timeout = 120
keepalive = 5
max_requests = 200
max_requests_jitter = 20
reload = True

# Server socket configuration
backlog = 2048

# Logging configuration
accesslog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'
loglevel = "info"
capture_output = True
enable_stdio_inheritance = True
errorlog = "-"

# Process naming
proc_name = 'aitherapist'

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Resource limits
limit_request_line = 8190
limit_request_fields = 100
limit_request_field_size = 8190

# Timeouts
graceful_timeout = 120
keep_alive = 5

# Server hooks
def on_starting(server):
    server.log.info("Starting AITherapist server...")

def on_reload(server):
    server.log.info("Reloading AITherapist server...")

def on_exit(server):
    server.log.info("Shutting down AITherapist server...")
