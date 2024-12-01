import multiprocessing
import os

# Worker configuration
workers = 1  # Reduced for better stability with async operations
worker_class = 'gevent'
worker_connections = 1000
timeout = 300  # Increased timeout for longer conversations
keepalive = 5
max_requests = 0  # Disabled to prevent worker recycling
max_requests_jitter = 0

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
