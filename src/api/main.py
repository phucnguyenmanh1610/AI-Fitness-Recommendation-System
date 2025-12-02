"""
FastAPI main application
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from contextlib import asynccontextmanager

from .config import settings
from .routes import health, workout, meal

# Setup logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for startup and shutdown"""
    # Startup
    logger.info("Starting AI Fitness & Health Recommendation System API...")
    logger.info(f"API Version: {settings.API_VERSION}")
    logger.info(f"Debug Mode: {settings.DEBUG}")
    
    # Load models on startup (optional - can be lazy loaded)
    try:
        # Models will be loaded on first request
        logger.info("Models will be loaded on first request")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="AI-powered fitness and health recommendation system with ML predictions",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix=settings.API_PREFIX, tags=["Health Prediction"])
app.include_router(workout.router, prefix=settings.API_PREFIX, tags=["Workout Recommendation"])
app.include_router(meal.router, prefix=settings.API_PREFIX, tags=["Meal Recommendation"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Fitness & Health Recommendation System API",
        "version": settings.API_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AI Fitness API",
        "version": settings.API_VERSION
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "message": str(exc) if settings.DEBUG else "An error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.MAX_WORKERS
    )

