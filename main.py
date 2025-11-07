# main.py (updated)
import sqlite3
import logging
from typing import Dict
from contextlib import contextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel, EmailStr, field_validator, ConfigDict
from pathlib import Path
import datetime
import os
import re
import threading
from email_validator import validate_email, EmailNotValidError

# -------------------------
# Logging configuration
# -------------------------
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG while developing
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------------
# Paths and environment
# -------------------------
APP_DIR = Path(__file__).parent.resolve()
DB_PATH = APP_DIR / "data.db"
STATIC_DIR = APP_DIR / "static"

# Ensure static dir exists before mounting
STATIC_DIR.mkdir(parents=True, exist_ok=True)

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000"
).split(",")

# -------------------------
# FastAPI app & middleware
# -------------------------
app = FastAPI(
    title="Master AI Landing - API",
    description="Backend API for Master AI educational platform",
    version="1.0.0",
    docs_url="/docs" if ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if ENVIRONMENT == "development" else None
)

# Security middleware (production-only)
if ENVIRONMENT == "production":
    app.add_middleware(HTTPSRedirectMiddleware)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Security helper (placeholder)
security = HTTPBearer(auto_error=False)

# Mount static files (directory exists because we created it above)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# -------------------------
# Database helpers
# -------------------------
@contextmanager
def get_conn():
    """Yield a sqlite3 connection with sane defaults and cleanup."""
    conn = None
    try:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        # Improve concurrency and enforce foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        logger.debug("Database connection established")
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}", exc_info=True)
        # Convert DB errors into 503 for clients
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database temporarily unavailable"
        )
    finally:
        if conn:
            conn.close()
            logger.debug("Database connection closed")

def init_db():
    """Create tables and indexes if missing."""
    logger.info("Initializing database...")
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS waitlist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS enrollments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    full_name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    track TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    status TEXT DEFAULT 'pending'
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    endpoint TEXT NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    referer TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            # Indexes for performance
            cur.execute("CREATE INDEX IF NOT EXISTS idx_waitlist_email ON waitlist(email)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_waitlist_created ON waitlist(created_at)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_enrollments_email ON enrollments(email)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_enrollments_status ON enrollments(status)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_analytics_created ON analytics(created_at)")
            conn.commit()
            logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        raise

def check_database_state():
    """Log tables and row counts (safe check)."""
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cur.fetchall()
            table_names = [table['name'] for table in tables]
            logger.info(f"Database tables: {table_names}")
            for table in ['waitlist', 'enrollments', 'analytics']:
                if table in table_names:
                    cur.execute(f"SELECT COUNT(*) as count FROM {table}")
                    count = cur.fetchone()['count']
                    logger.info(f"Table {table}: {count} rows")
                else:
                    logger.warning(f"Table {table} does not exist!")
    except Exception as e:
        logger.error(f"Database state check failed: {e}", exc_info=True)

# -------------------------
# Startup / Shutdown
# -------------------------
@app.on_event("startup")
def startup():
    logger.info(f"Starting Master AI API in {ENVIRONMENT} environment")
    init_db()
    check_database_state()
    logger.info("Application startup completed")

@app.on_event("shutdown")
def shutdown():
    logger.info("Shutting down Master AI API")

# -------------------------
# Pydantic models (V2-style)
# -------------------------
class WaitlistIn(BaseModel):
    email: EmailStr

    @field_validator('email')
    @classmethod
    def validate_email_domain(cls, v: str) -> str:
        try:
            email_info = validate_email(v, check_deliverability=False)
            return email_info.normalized
        except EmailNotValidError as e:
            raise ValueError(str(e))

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"email": "student@example.com"}
        }
    )

class EnrollIn(BaseModel):
    name: str
    email: EmailStr
    track: str

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2 or len(v) > 120:
            raise ValueError('Name must be between 2 and 120 characters')
        if not re.match(r'^[a-zA-Z\s\-\.\']+$', v):
            raise ValueError('Name contains invalid characters')
        return v.title()

    @field_validator('email')
    @classmethod
    def validate_email_domain(cls, v: str) -> str:
        try:
            email_info = validate_email(v, check_deliverability=False)
            return email_info.normalized
        except EmailNotValidError as e:
            raise ValueError(str(e))

    @field_validator('track')
    @classmethod
    def validate_track(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3 or len(v) > 60:
            raise ValueError('Track must be between 3 and 60 characters')
        valid_tracks = ["Data Foundations", "Machine Learning", "Deep Learning & MLOps"]
        if v not in valid_tracks:
            raise ValueError(f"Track must be one of: {', '.join(valid_tracks)}")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "John Doe",
                "email": "john@example.com",
                "track": "Machine Learning"
            }
        }
    )

# -------------------------
# Rate limiter (thread-safe) 
# -------------------------
class RateLimiter:
    def __init__(self, requests_per_minute: int = 20):
        self.requests_per_minute = requests_per_minute
        self.requests = {}
        self.lock = threading.Lock()

    def is_rate_limited(self, identifier: str) -> bool:
        """Thread-safe minute-based rate limiter."""
        try:
            with self.lock:
                now = datetime.datetime.utcnow()
                minute_key = now.strftime("%Y-%m-%dT%H:%M")
                key = f"{identifier}:{minute_key}"
                self.requests.setdefault(key, 0)
                self.requests[key] += 1
                logger.debug(f"Rate limit check: {key} = {self.requests[key]}")

                # Basic cleanup: remove keys older than current minute
                current_minute = minute_key
                keys_to_delete = [k for k in list(self.requests.keys())
                                  if ":" in k and k.split(":", 1)[1] < current_minute]
                for k in keys_to_delete:
                    del self.requests[k]
                    logger.debug(f"Cleaned old rate limit key: {k}")

                return self.requests[key] > self.requests_per_minute
        except Exception as e:
            logger.error(f"Rate limiter error: {e}", exc_info=True)
            return False

rate_limiter = RateLimiter(requests_per_minute=20)

# -------------------------
# Helpers
# -------------------------
def get_client_info(request: Request) -> Dict[str, str]:
    """Safely extract client info and fallback for proxied requests."""
    x_forwarded = request.headers.get("x-forwarded-for")
    ip = None
    if x_forwarded:
        ip = x_forwarded.split(",")[0].strip()
    else:
        try:
            ip = request.client.host if request.client else "unknown"
        except Exception:
            ip = "unknown"

    client_info = {
        "ip_address": ip,
        "user_agent": request.headers.get("user-agent", "unknown"),
        "referer": request.headers.get("referer")
    }
    logger.debug(f"Client info: {client_info}")
    return client_info

# -------------------------
# Routes
# -------------------------
@app.get("/", response_class=FileResponse)
async def index(request: Request):
    """Serve landing page and track analytics (non-blocking to client)."""
    client_info = get_client_info(request)

    # Track page view (best-effort)
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            now = datetime.datetime.utcnow().isoformat()
            cur.execute(
                "INSERT INTO analytics (endpoint, ip_address, user_agent, referer, created_at) VALUES (?, ?, ?, ?, ?)",
                ("/", client_info["ip_address"], client_info["user_agent"], client_info["referer"], now)
            )
            conn.commit()
            logger.debug("Page view tracked in analytics")
    except Exception as e:
        # Log but don't fail the request
        logger.error(f"Failed to track analytics: {e}", exc_info=True)

    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        logger.error("index.html not found in static directory")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Landing page not available")

    # FileResponse handles Content-Length correctly by streaming the file
    return FileResponse(str(index_file), media_type="text/html")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve favicon or return an empty 204 response correctly."""
    favicon_path = STATIC_DIR / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(str(favicon_path))
    # Return a bare Response with no body for 204 (no content)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@app.post("/api/waitlist")
async def join_waitlist(payload: WaitlistIn, request: Request):
    """Add an email to the waitlist with rate limiting and duplicate handling."""
    logger.info(f"Waitlist request received: {payload.email}")
    client_info = get_client_info(request)

    try:
        if rate_limiter.is_rate_limited(client_info["ip_address"]):
            logger.warning(f"Rate limit exceeded for IP: {client_info['ip_address']}")
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many requests.")
    except Exception as e:
        logger.error(f"Rate limiter error: {e}", exc_info=True)

    try:
        with get_conn() as conn:
            cur = conn.cursor()
            now = datetime.datetime.utcnow().isoformat()
            cur.execute("SELECT id FROM waitlist WHERE email = ?", (payload.email.lower(),))
            existing = cur.fetchone()
            if existing:
                logger.info(f"Waitlist duplicate attempt for email: {payload.email}")
                return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "Already on waitlist"})
            cur.execute(
                """INSERT INTO waitlist (email, created_at, ip_address, user_agent) VALUES (?, ?, ?, ?)""",
                (payload.email.lower(), now, client_info["ip_address"], client_info["user_agent"])
            )
            conn.commit()
            logger.info(f"New waitlist signup saved: {payload.email}")
            return JSONResponse(status_code=status.HTTP_201_CREATED, content={"message": "Successfully added to waitlist"})
    except sqlite3.IntegrityError as e:
        # Unique constraint may raise here concurrently
        logger.warning(f"Integrity error while inserting waitlist: {e}", exc_info=True)
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "Already on waitlist"})
    except sqlite3.Error as e:
        logger.error(f"Database error in waitlist: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service temporarily unavailable")

@app.post("/api/enroll")
async def enroll(payload: EnrollIn, request: Request):
    """Process enrollment request with validation and tracking."""
    logger.info(f"Enrollment request received: {payload.name}, {payload.email}, {payload.track}")
    client_info = get_client_info(request)

    try:
        if rate_limiter.is_rate_limited(client_info["ip_address"]):
            logger.warning(f"Rate limit exceeded for IP: {client_info['ip_address']}")
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many requests.")
    except Exception as e:
        logger.error(f"Rate limiter error: {e}", exc_info=True)

    try:
        with get_conn() as conn:
            cur = conn.cursor()
            now = datetime.datetime.utcnow().isoformat()
            cur.execute("SELECT id FROM enrollments WHERE email = ? AND track = ?", (payload.email.lower(), payload.track))
            existing = cur.fetchone()
            if existing:
                logger.info(f"Duplicate enrollment attempt: {payload.email} for {payload.track}")
                return JSONResponse(status_code=status.HTTP_200_OK, content={"message": f"You're already enrolled in {payload.track}"})
            cur.execute(
                """INSERT INTO enrollments (full_name, email, track, created_at, ip_address, user_agent) VALUES (?, ?, ?, ?, ?, ?)""",
                (payload.name, payload.email.lower(), payload.track, now, client_info["ip_address"], client_info["user_agent"])
            )
            conn.commit()
            logger.info(f"New enrollment saved: {payload.name} ({payload.email}) for {payload.track}")
            return JSONResponse(
                status_code=status.HTTP_201_CREATED,
                content={
                    "message": f"Thanks {payload.name.split()[0]} — we've saved your application and sent next steps to {payload.email}",
                    "next_steps": "Check your email for confirmation and next steps"
                }
            )
    except sqlite3.IntegrityError as e:
        logger.warning(f"Integrity error during enrollment insert: {e}", exc_info=True)
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "You're already enrolled"})
    except sqlite3.Error as e:
        logger.error(f"Database error in enrollment: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service temporarily unavailable")

# -------------------------
# Debug endpoints
# -------------------------
@app.get("/api/debug/waitlist")
async def debug_waitlist():
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM waitlist ORDER BY created_at DESC")
            rows = cur.fetchall()
            data = [dict(row) for row in rows]
            return {"count": len(data), "data": data}
    except Exception as e:
        logger.error(f"Debug error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/debug/enrollments")
async def debug_enrollments():
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM enrollments ORDER BY created_at DESC")
            rows = cur.fetchall()
            data = [dict(row) for row in rows]
            return {"count": len(data), "data": data}
    except Exception as e:
        logger.error(f"Debug error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/debug/tables")
async def debug_tables():
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cur.fetchall()
            result = {}
            for table in tables:
                table_name = table['name']
                cur.execute(f"SELECT COUNT(*) as count FROM {table_name}")
                count = cur.fetchone()['count']
                cur.execute(f"SELECT * FROM {table_name} LIMIT 5")
                sample_data = [dict(row) for row in cur.fetchall()]
                result[table_name] = {"count": count, "sample_data": sample_data}
            return result
    except Exception as e:
        logger.error(f"Debug tables error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

# -------------------------
# Stats & health
# -------------------------
@app.get("/api/stats")
async def get_stats():
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) as count FROM waitlist")
            waitlist_count = cur.fetchone()["count"]
            cur.execute("SELECT track, COUNT(*) as count FROM enrollments GROUP BY track")
            enrollment_stats = {row["track"]: row["count"] for row in cur.fetchall()}
            total_enrollments = sum(enrollment_stats.values()) if enrollment_stats else 0
            week_ago = (datetime.datetime.utcnow() - datetime.timedelta(days=7)).isoformat()
            cur.execute("SELECT COUNT(*) as count FROM waitlist WHERE created_at > ?", (week_ago,))
            recent_waitlist = cur.fetchone()["count"]
            cur.execute("SELECT COUNT(*) as count FROM enrollments WHERE created_at > ?", (week_ago,))
            recent_enrollments = cur.fetchone()["count"]
            return {
                "waitlist_total": waitlist_count,
                "enrollments_total": total_enrollments,
                "enrollments_by_track": enrollment_stats,
                "recent_activity": {
                    "waitlist_last_7_days": recent_waitlist,
                    "enrollments_last_7_days": recent_enrollments
                }
            }
    except sqlite3.Error as e:
        logger.error(f"Database error in stats: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service temporarily unavailable")

@app.get("/api/health")
async def health():
    health_status = {
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "environment": ENVIRONMENT,
        "version": "1.0.0"
    }
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            health_status["database"] = "connected"
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row['name'] for row in cur.fetchall()]
            health_status["tables"] = tables
    except Exception as e:
        health_status["database"] = "disconnected"
        health_status["status"] = "degraded"
        logger.error(f"Health check failed: {e}", exc_info=True)
    return health_status

# -------------------------
# Error handlers
# -------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTPException: {exc.status_code} - {exc.detail}")
    # Always return a JSON object body for HTTP exceptions
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    # Provide a stable, predictable JSON response body
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": "Internal server error"})

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=ENVIRONMENT == "development",
        log_level="debug"
    )
