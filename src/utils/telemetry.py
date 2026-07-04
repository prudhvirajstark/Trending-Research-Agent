"""
Logging and telemetry utilities.
"""

import logging
import os
import json
from datetime import datetime


def setup_logging(log_level: str = "INFO", log_format: str = "text") -> None:
    """
    Configure logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format ("text" or "json")
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    if log_format == "json":
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)

    # Optional file handler
    if os.getenv("ENABLE_FILE_LOGGING") == "true":
        file_handler = logging.FileHandler("phd_thesis_agent.log")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class ExecutionTracer:
    """Traces execution of agents for observability."""

    def __init__(self):
        """Initialize execution tracer."""
        self.traces = []

    def add_trace(
        self,
        agent_name: str,
        operation: str,
        start_time: datetime,
        end_time: datetime,
        metadata: dict = None,
    ) -> None:
        """
        Add an execution trace.

        Args:
            agent_name: Name of the agent
            operation: Operation performed
            start_time: Start time of operation
            end_time: End time of operation
            metadata: Optional metadata dictionary
        """
        duration = (end_time - start_time).total_seconds()

        trace = {
            "timestamp": start_time.isoformat(),
            "agent": agent_name,
            "operation": operation,
            "duration_seconds": duration,
            "metadata": metadata or {},
        }

        self.traces.append(trace)

        logger = logging.getLogger(f"trace.{agent_name}")
        logger.info(f"{operation} completed in {duration:.2f}s", extra=trace)

    def get_traces(self) -> list:
        """Get all recorded traces."""
        return self.traces

    def clear_traces(self) -> None:
        """Clear all traces."""
        self.traces = []

    def get_metrics(self) -> dict:
        """
        Get metrics from traces.

        Returns:
            Dictionary with aggregate metrics
        """
        if not self.traces:
            return {}

        total_time = sum(t["duration_seconds"] for t in self.traces)
        avg_time = total_time / len(self.traces) if self.traces else 0

        agents = {}
        for trace in self.traces:
            agent = trace["agent"]
            if agent not in agents:
                agents[agent] = {"count": 0, "total_time": 0}
            agents[agent]["count"] += 1
            agents[agent]["total_time"] += trace["duration_seconds"]

        return {
            "total_executions": len(self.traces),
            "total_time_seconds": total_time,
            "average_time_seconds": avg_time,
            "by_agent": agents,
        }
