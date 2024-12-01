import multiprocessing
import os

# Worker configuration
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'gevent'
worker_connections = 1000
timeout = 120
keepalive = 5
max_requests = 1000
max_requests_jitter = 50

# Server socket configuration
bind = "0.0.0.0:" + str(os.getenv("PORT", 8000))
backlog = 2048

# Logging configuration
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = 'aitherapist'

# SSL configuration (if needed)
# keyfile = 'path/to/keyfile'
# certfile = 'path/to/certfile'

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Server hooks
def on_starting(server):
    server.log.info("Starting AITherapist server...")

def on_reload(server):
    server.log.info("Reloading AITherapist server...")

def on_exit(server):
    server.log.info("Shutting down AITherapist server...")
