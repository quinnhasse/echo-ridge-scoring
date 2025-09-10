"""
FastAPI application for Echo Ridge scoring service.

Provides REST endpoints for AI-readiness scoring of SMB companies.
"""

from contextlib import asynccontextmanager
from typing import Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .dependencies import set_persistence_manager
from ..echo_ridge_scoring.persistence import PersistenceManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Initialize persistence manager on startup
    persistence_manager = PersistenceManager()
    
    # Set global persistence manager for dependencies
    set_persistence_manager(persistence_manager)
    
    yield
    
    # Cleanup on shutdown
    if persistence_manager:
        await persistence_manager.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Echo Ridge Scoring API",
        description="""
        AI-Readiness scoring service for SMB companies.
        
        Provides deterministic scoring based on Digital, Operations, Information Flow, 
        Market, and Budget dimensions with integrated risk assessment and feasibility gates.
        """,
        version="1.1.0",
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import router here to avoid circular imports
    from .endpoints import router
    
    # Include API routes
    app.include_router(router, prefix="", tags=["scoring"])

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        """Custom HTTP exception handler with structured error responses."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.status_code,
                    "message": exc.detail,
                    "type": "HTTPException"
                }
            }
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        """General exception handler for unhandled exceptions."""
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": 500,
                    "message": "Internal server error",
                    "type": "InternalServerError"
                }
            }
        )

    return app


# Create app instance
app = create_app()





if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)