import multiprocessing
import os

# Worker configuration
port = int(os.getenv('PORT', 10000))  # Use Render's PORT env variable
bind = f"0.0.0.0:{port}"
worker_class = "sync"
workers = 2
threads = 1
timeout = 120  # Increased timeout
keepalive = 5
max_requests = 100
max_requests_jitter = 20
preload_app = True
reload = False

# Server socket configuration
backlog = 2048

# Logging configuration
accesslog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'
loglevel = "info"  # Changed to info for better debugging
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
worker_tmp_dir = '/tmp'  # Changed from /dev/shm for compatibility

# Resource limits
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Timeouts
graceful_timeout = 30  # Increased timeout
keep_alive = 5

# Server hooks
def on_starting(server):
    server.log.info(f"Starting AITherapist server on port {port}...")

def on_reload(server):
    server.log.info("Reloading AITherapist server...")

def on_exit(server):
    server.log.info("Shutting down AITherapist server...")
