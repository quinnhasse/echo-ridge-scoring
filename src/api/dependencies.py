"""
FastAPI dependencies for the Echo Ridge scoring service.
"""

from fastapi import HTTPException, status

from ..batch import BatchProcessor
from ..normalization import NormContext
from ..persistence import PersistenceManager


# Global persistence manager instance
_persistence_manager: PersistenceManager = None


def set_persistence_manager(manager: PersistenceManager):
    """Set the global persistence manager instance."""
    global _persistence_manager
    _persistence_manager = manager


def get_persistence_manager() -> PersistenceManager:
    """Get the global persistence manager instance."""
    if _persistence_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Persistence manager not initialized"
        )
    return _persistence_manager


async def get_batch_processor() -> BatchProcessor:
    """Dependency to get a configured BatchProcessor instance."""
    try:
        persistence_manager = get_persistence_manager()
        processor = BatchProcessor(
            persistence_manager=persistence_manager
        )
        return processor
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to initialize batch processor: {str(e)}"
        )


async def get_norm_context() -> NormContext:
    """Dependency to get the current normalization context."""
    try:
        persistence_manager = get_persistence_manager()
        
        # Try to load the most recent norm context from database
        context = persistence_manager.get_latest_norm_context()
        
        if context is None:
            # If no context exists, create a default one
            # This would typically happen on first service startup
            context = NormContext()
            context.fit_defaults()  # Use default normalization parameters
            
        return context
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to load normalization context: {str(e)}"
        )