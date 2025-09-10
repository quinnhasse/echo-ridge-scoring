"""
Monitoring and observability module for Echo Ridge scoring engine.

Provides structured JSON logging, performance metrics, error tracking,
and operational monitoring capabilities for production deployments.
"""

import json
import time
import uuid
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, asdict

import structlog
from pydantic import BaseModel, Field


class LogLevel(str, Enum):
    """Standard log levels."""
    DEBUG = "debug"
    INFO = "info" 
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ComponentType(str, Enum):
    """System components for structured logging."""
    API = "api"
    SCORING_ENGINE = "scoring_engine"
    BATCH_PROCESSOR = "batch_processor"
    CALIBRATION = "calibration"
    DRIFT_DETECTION = "drift_detection"
    PERSISTENCE = "persistence"
    NORMALIZATION = "normalization"


class EventType(str, Enum):
    """Types of events to log."""
    REQUEST_START = "request_start"
    REQUEST_END = "request_end"
    SCORING_START = "scoring_start"
    SCORING_END = "scoring_end"
    ERROR = "error"
    PERFORMANCE_ALERT = "performance_alert"
    DRIFT_ALERT = "drift_alert"
    CALIBRATION_UPDATE = "calibration_update"
    SYSTEM_HEALTH = "system_health"


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    
    operation_name: str
    start_time: float
    end_time: float
    duration_ms: float
    success: bool
    error_message: Optional[str] = None
    
    # Request-specific metrics
    request_id: Optional[str] = None
    company_id: Optional[str] = None
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    
    # System resource metrics
    memory_usage_mb: Optional[float] = None
    cpu_usage_pct: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


class StructuredLogger:
    """
    Structured logger with JSON output and correlation ID support.
    
    Provides consistent logging format across all Echo Ridge components
    with automatic correlation tracking and performance monitoring.
    """
    
    def __init__(self, 
                 component: ComponentType,
                 service_name: str = "echo-ridge-scoring",
                 version: str = "1.0.0"):
        """
        Initialize structured logger.
        
        Args:
            component: Which component is doing the logging.
            service_name: Name of the service for log aggregation.
            version: Service version for tracking deployments.
        """
        self.component = component
        self.service_name = service_name
        self.version = version
        
        # Configure structlog for JSON output
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                self._add_context,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        self.logger = structlog.get_logger()
        self._correlation_id = None
        
    def _add_context(self, _, __, event_dict):
        """Add standard context to all log events."""
        event_dict.update({
            "service": self.service_name,
            "version": self.version,
            "component": self.component.value,
            "correlation_id": self._correlation_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        return event_dict
    
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for request tracking."""
        self._correlation_id = correlation_id
        
    def generate_correlation_id(self) -> str:
        """Generate new correlation ID."""
        self._correlation_id = str(uuid.uuid4())
        return self._correlation_id
    
    def info(self, message: str, **kwargs):
        """Log info level message."""
        self.logger.info(message, **kwargs)
        
    def warning(self, message: str, **kwargs):
        """Log warning level message."""
        self.logger.warning(message, **kwargs)
        
    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error level message."""
        if error:
            kwargs.update({
                "error_type": type(error).__name__,
                "error_message": str(error),
                "error_traceback": traceback.format_exc()
            })
        self.logger.error(message, **kwargs)
        
    def debug(self, message: str, **kwargs):
        """Log debug level message."""
        self.logger.debug(message, **kwargs)
        
    def log_event(self, 
                  event_type: EventType, 
                  message: str,
                  level: LogLevel = LogLevel.INFO,
                  **context):
        """Log structured event with type and context."""
        log_data = {
            "event_type": event_type.value,
            "message": message,
            **context
        }
        
        if level == LogLevel.DEBUG:
            self.logger.debug(message, **log_data)
        elif level == LogLevel.INFO:
            self.logger.info(message, **log_data) 
        elif level == LogLevel.WARNING:
            self.logger.warning(message, **log_data)
        elif level == LogLevel.ERROR:
            self.logger.error(message, **log_data)
        elif level == LogLevel.CRITICAL:
            self.logger.critical(message, **log_data)
    
    def log_performance(self, metrics: PerformanceMetrics):
        """Log performance metrics."""
        self.log_event(
            EventType.REQUEST_END if metrics.operation_name.startswith('request') else EventType.SCORING_END,
            f"Operation {metrics.operation_name} completed",
            level=LogLevel.INFO,
            **metrics.to_dict()
        )
    
    def log_request_start(self, 
                         endpoint: str, 
                         method: str,
                         request_id: Optional[str] = None,
                         **context):
        """Log request start event."""
        if not request_id:
            request_id = self.generate_correlation_id()
        else:
            self.set_correlation_id(request_id)
            
        self.log_event(
            EventType.REQUEST_START,
            f"{method} {endpoint} started",
            endpoint=endpoint,
            method=method,
            request_id=request_id,
            **context
        )
        
        return request_id
    
    def log_request_end(self,
                       endpoint: str,
                       method: str, 
                       status_code: int,
                       duration_ms: float,
                       **context):
        """Log request completion event."""
        level = LogLevel.ERROR if status_code >= 400 else LogLevel.INFO
        
        self.log_event(
            EventType.REQUEST_END,
            f"{method} {endpoint} completed",
            level=level,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            duration_ms=duration_ms,
            **context
        )


class PerformanceMonitor:
    """
    Performance monitoring and alerting for scoring operations.
    
    Tracks latency, throughput, error rates, and generates alerts
    when performance thresholds are exceeded.
    """
    
    def __init__(self, 
                 logger: StructuredLogger,
                 latency_p99_threshold_ms: float = 150.0,
                 error_rate_threshold: float = 0.01):
        """
        Initialize performance monitor.
        
        Args:
            logger: Structured logger instance.
            latency_p99_threshold_ms: p99 latency threshold in milliseconds.
            error_rate_threshold: Error rate threshold (0.01 = 1%).
        """
        self.logger = logger
        self.latency_p99_threshold_ms = latency_p99_threshold_ms
        self.error_rate_threshold = error_rate_threshold
        
        # In-memory metrics (in production, would use proper metrics store)
        self.recent_latencies: List[float] = []
        self.recent_errors: List[bool] = []
        self.max_history_size = 1000
    
    def record_operation(self, 
                        operation_name: str,
                        duration_ms: float,
                        success: bool,
                        **context) -> PerformanceMetrics:
        """
        Record operation performance metrics.
        
        Args:
            operation_name: Name of the operation.
            duration_ms: Operation duration in milliseconds.
            success: Whether operation succeeded.
            **context: Additional context for logging.
            
        Returns:
            PerformanceMetrics instance.
        """
        # Create metrics object
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=time.time() - duration_ms/1000,
            end_time=time.time(),
            duration_ms=duration_ms,
            success=success,
            **context
        )
        
        # Update history
        self.recent_latencies.append(duration_ms)
        self.recent_errors.append(not success)
        
        # Trim history to prevent memory growth
        if len(self.recent_latencies) > self.max_history_size:
            self.recent_latencies = self.recent_latencies[-self.max_history_size:]
            self.recent_errors = self.recent_errors[-self.max_history_size:]
        
        # Check for performance alerts
        self._check_performance_alerts(metrics)
        
        # Log metrics
        self.logger.log_performance(metrics)
        
        return metrics
    
    def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check if performance alerts should be triggered."""
        
        # Check individual operation latency
        if metrics.duration_ms > self.latency_p99_threshold_ms:
            self.logger.log_event(
                EventType.PERFORMANCE_ALERT,
                f"High latency detected for {metrics.operation_name}",
                level=LogLevel.WARNING,
                duration_ms=metrics.duration_ms,
                threshold_ms=self.latency_p99_threshold_ms,
                operation_name=metrics.operation_name
            )
        
        # Check recent error rate (if we have enough data)
        if len(self.recent_errors) >= 10:
            recent_error_rate = sum(self.recent_errors[-10:]) / 10
            
            if recent_error_rate > self.error_rate_threshold:
                self.logger.log_event(
                    EventType.PERFORMANCE_ALERT,
                    f"High error rate detected",
                    level=LogLevel.WARNING,
                    error_rate=recent_error_rate,
                    threshold=self.error_rate_threshold,
                    recent_operations=len(self.recent_errors)
                )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary statistics."""
        if not self.recent_latencies:
            return {"status": "no_data"}
        
        import numpy as np
        
        latencies = np.array(self.recent_latencies)
        errors = np.array(self.recent_errors)
        
        return {
            "total_operations": len(self.recent_latencies),
            "latency_stats": {
                "mean_ms": float(np.mean(latencies)),
                "median_ms": float(np.median(latencies)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "p99_ms": float(np.percentile(latencies, 99)),
                "max_ms": float(np.max(latencies))
            },
            "error_stats": {
                "total_errors": int(np.sum(errors)),
                "error_rate": float(np.mean(errors)),
                "success_rate": float(1 - np.mean(errors))
            },
            "thresholds": {
                "latency_p99_threshold_ms": self.latency_p99_threshold_ms,
                "error_rate_threshold": self.error_rate_threshold
            }
        }


def monitor_performance(operation_name: str, 
                       logger: StructuredLogger,
                       performance_monitor: Optional[PerformanceMonitor] = None):
    """
    Decorator to monitor function performance.
    
    Args:
        operation_name: Name of the operation being monitored.
        logger: Structured logger instance.
        performance_monitor: Optional performance monitor instance.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error = None
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                return result
                
            except Exception as e:
                success = False
                error = e
                logger.error(
                    f"Operation {operation_name} failed",
                    error=e,
                    operation_name=operation_name
                )
                raise
                
            finally:
                # Record performance
                duration_ms = (time.time() - start_time) * 1000
                
                if performance_monitor:
                    performance_monitor.record_operation(
                        operation_name=operation_name,
                        duration_ms=duration_ms,
                        success=success,
                        error_message=str(error) if error else None
                    )
                else:
                    # Just log without monitoring
                    logger.log_event(
                        EventType.SCORING_END,
                        f"Operation {operation_name} {'succeeded' if success else 'failed'}",
                        level=LogLevel.INFO if success else LogLevel.ERROR,
                        operation_name=operation_name,
                        duration_ms=duration_ms,
                        success=success
                    )
        
        return wrapper
    return decorator


@contextmanager
def performance_context(operation_name: str,
                       logger: StructuredLogger,
                       performance_monitor: Optional[PerformanceMonitor] = None,
                       **context):
    """
    Context manager for monitoring performance of code blocks.
    
    Args:
        operation_name: Name of the operation being monitored.
        logger: Structured logger instance.
        performance_monitor: Optional performance monitor instance.
        **context: Additional context to include in logs.
    """
    start_time = time.time()
    success = True
    error = None
    
    logger.log_event(
        EventType.SCORING_START,
        f"Operation {operation_name} started",
        operation_name=operation_name,
        **context
    )
    
    try:
        yield
        
    except Exception as e:
        success = False
        error = e
        logger.error(
            f"Operation {operation_name} failed",
            error=e,
            operation_name=operation_name,
            **context
        )
        raise
        
    finally:
        duration_ms = (time.time() - start_time) * 1000
        
        if performance_monitor:
            performance_monitor.record_operation(
                operation_name=operation_name,
                duration_ms=duration_ms,
                success=success,
                error_message=str(error) if error else None,
                **context
            )
        else:
            logger.log_event(
                EventType.SCORING_END,
                f"Operation {operation_name} {'succeeded' if success else 'failed'}",
                level=LogLevel.INFO if success else LogLevel.ERROR,
                operation_name=operation_name,
                duration_ms=duration_ms,
                success=success,
                **context
            )


class HealthChecker:
    """
    System health monitoring for Echo Ridge components.
    
    Performs health checks on various system components and
    provides consolidated health status.
    """
    
    def __init__(self, logger: StructuredLogger):
        """Initialize health checker."""
        self.logger = logger
    
    def check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            from .persistence import PersistenceManager
            
            # Basic connectivity test
            pm = PersistenceManager()
            start_time = time.time()
            
            # Try a simple operation
            pm.get_latest_norm_context()
            
            duration_ms = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "response_time_ms": duration_ms,
                "details": "Database connectivity OK"
            }
            
        except Exception as e:
            return {
                "status": "unhealthy", 
                "error": str(e),
                "details": "Database connectivity failed"
            }
    
    def check_scoring_engine_health(self) -> Dict[str, Any]:
        """Check scoring engine functionality."""
        try:
            from .scoring import SubscoreCalculator, FinalScorer
            from .normalization import NormContext
            from .schema import CompanySchema
            
            # Test with minimal valid data
            test_company_data = {
                "company_id": "health_check",
                "domain": "test.com",
                "digital": {"website_score": 80},
                "ops": {"employee_count": 10, "years_in_business": 2},
                "info_flow": {"data_integration_score": 70},
                "market": {"market_size_score": 60, "competition_level": 50},
                "budget": {"revenue_est_usd": 500000, "tech_budget_pct": 5},
                "meta": {"source": "health_check", "source_confidence": 0.8}
            }
            
            start_time = time.time()
            
            # Test scoring pipeline
            company = CompanySchema(**test_company_data)
            norm_context = NormContext()
            norm_context.fit_defaults()
            
            calc = SubscoreCalculator(norm_context)
            subscores = calc.calculate_subscores(company)
            
            scorer = FinalScorer({
                'digital': 0.25, 'operations': 0.20, 'info_flow': 0.20,
                'market': 0.20, 'budget': 0.15
            })
            final_score = scorer.calculate_final_score(subscores)
            
            duration_ms = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "response_time_ms": duration_ms,
                "test_score": final_score.final_score,
                "details": "Scoring engine functional"
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "details": "Scoring engine test failed"
            }
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run comprehensive health checks."""
        health_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": "healthy",
            "components": {}
        }
        
        # Check each component
        checks = [
            ("database", self.check_database_health),
            ("scoring_engine", self.check_scoring_engine_health)
        ]
        
        unhealthy_components = []
        
        for component_name, check_func in checks:
            try:
                result = check_func()
                health_results["components"][component_name] = result
                
                if result["status"] != "healthy":
                    unhealthy_components.append(component_name)
                    
            except Exception as e:
                health_results["components"][component_name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "details": f"Health check failed for {component_name}"
                }
                unhealthy_components.append(component_name)
        
        # Determine overall status
        if unhealthy_components:
            if len(unhealthy_components) >= len(checks) / 2:
                health_results["overall_status"] = "unhealthy"
            else:
                health_results["overall_status"] = "degraded"
                
            health_results["unhealthy_components"] = unhealthy_components
        
        # Log health check results
        self.logger.log_event(
            EventType.SYSTEM_HEALTH,
            f"System health check completed: {health_results['overall_status']}",
            level=LogLevel.INFO if health_results["overall_status"] == "healthy" else LogLevel.WARNING,
            **health_results
        )
        
        return health_results


# Global logger instances for easy access
api_logger = StructuredLogger(ComponentType.API)
scoring_logger = StructuredLogger(ComponentType.SCORING_ENGINE) 
batch_logger = StructuredLogger(ComponentType.BATCH_PROCESSOR)
calibration_logger = StructuredLogger(ComponentType.CALIBRATION)
drift_logger = StructuredLogger(ComponentType.DRIFT_DETECTION)

# Global performance monitors
api_performance_monitor = PerformanceMonitor(api_logger)
scoring_performance_monitor = PerformanceMonitor(scoring_logger)

# Global health checker
health_checker = HealthChecker(api_logger)