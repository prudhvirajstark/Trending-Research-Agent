"""
Rate limiter for Gemini API free tier compliance

Free tier limits:
- 60 requests per minute
- 1.5M tokens per day
- No concurrent requests limit, but practical rate limiting recommended

This module provides request throttling to stay within free tier limits.
"""

import asyncio
import time
import logging
from typing import Optional, Dict
from datetime import datetime, timedelta
from collections import deque


logger = logging.getLogger(__name__)


class RateLimiter:
    """Async rate limiter for API requests."""

    def __init__(
        self,
        requests_per_minute: int = 30,  # Conservative for free tier (60 max)
        min_delay_seconds: float = 2.0,  # Minimum delay between requests
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Max requests per minute (default 30, free tier max is 60)
            min_delay_seconds: Minimum delay between consecutive requests
        """
        self.requests_per_minute = requests_per_minute
        self.min_delay_seconds = min_delay_seconds
        self.last_request_time: Optional[float] = None
        self.request_times: deque = deque()  # Track last minute of requests
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        """
        Acquire permission to make a request.
        Blocks until rate limit allows the request.
        """
        async with self.lock:
            now = time.time()

            # Remove requests older than 1 minute
            while (
                self.request_times
                and self.request_times[0] < now - 60
            ):
                self.request_times.popleft()

            # Check if we've exceeded requests per minute
            if len(self.request_times) >= self.requests_per_minute:
                # Calculate wait time until oldest request expires
                wait_time = 60 - (now - self.request_times[0])
                logger.warning(
                    f"Rate limit approaching ({len(self.request_times)}/{self.requests_per_minute}). "
                    f"Waiting {wait_time:.1f}s..."
                )
                await asyncio.sleep(wait_time + 0.1)
                # Recursively try again after waiting
                await self.acquire()
                return

            # Check minimum delay between requests
            if self.last_request_time:
                elapsed = now - self.last_request_time
                if elapsed < self.min_delay_seconds:
                    wait_time = self.min_delay_seconds - elapsed
                    logger.debug(f"Rate limiter: waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                    now = time.time()

            # Record this request
            self.request_times.append(now)
            self.last_request_time = now

    def get_metrics(self) -> Dict[str, any]:
        """Get rate limiter metrics."""
        now = time.time()
        # Count requests in last minute
        requests_last_minute = sum(
            1 for t in self.request_times if t > now - 60
        )
        return {
            "requests_last_minute": requests_last_minute,
            "capacity": self.requests_per_minute,
            "utilization": requests_last_minute / self.requests_per_minute,
            "last_request_time": self.last_request_time,
            "min_delay_seconds": self.min_delay_seconds,
        }

    def reset(self) -> None:
        """Reset rate limiter state."""
        self.request_times.clear()
        self.last_request_time = None
        logger.info("Rate limiter reset")


class TokenBudgetTracker:
    """Track token usage against free tier daily budget."""

    def __init__(self, daily_limit: int = 1_500_000):
        """
        Initialize token budget tracker.

        Args:
            daily_limit: Daily token limit for free tier (default 1.5M)
        """
        self.daily_limit = daily_limit
        self.tokens_used_today = 0
        self.reset_time = self._get_reset_time()

    def _get_reset_time(self) -> datetime:
        """Get next reset time (midnight UTC)."""
        now = datetime.utcnow()
        tomorrow = now + timedelta(days=1)
        return tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)

    def add_tokens(self, count: int) -> None:
        """
        Add tokens to usage counter.

        Args:
            count: Number of tokens to add
        """
        # Check if day has reset
        if datetime.utcnow() >= self.reset_time:
            self.tokens_used_today = 0
            self.reset_time = self._get_reset_time()
            logger.info("Daily token budget reset")

        self.tokens_used_today += count

        usage_percent = (self.tokens_used_today / self.daily_limit) * 100
        remaining = self.daily_limit - self.tokens_used_today

        if usage_percent > 90:
            logger.warning(
                f"Token budget at {usage_percent:.1f}% "
                f"({self.tokens_used_today:,}/{self.daily_limit:,}). "
                f"{remaining:,} remaining."
            )
        elif usage_percent > 75:
            logger.info(
                f"Token usage: {usage_percent:.1f}% "
                f"({self.tokens_used_today:,}/{self.daily_limit:,})"
            )

    def get_remaining_budget(self) -> int:
        """Get remaining tokens for today."""
        # Check if day has reset
        if datetime.utcnow() >= self.reset_time:
            self.tokens_used_today = 0
            self.reset_time = self._get_reset_time()

        return self.daily_limit - self.tokens_used_today

    def get_metrics(self) -> Dict[str, any]:
        """Get token budget metrics."""
        remaining = self.get_remaining_budget()
        return {
            "tokens_used": self.tokens_used_today,
            "daily_limit": self.daily_limit,
            "remaining": remaining,
            "usage_percent": (self.tokens_used_today / self.daily_limit) * 100,
            "reset_time": self.reset_time.isoformat(),
        }
