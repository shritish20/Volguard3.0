"""
Rate Limiter Utility
Prevents 429 Errors from Upstox API
"""
import time
import threading
from functools import wraps
from typing import Dict, Deque
from collections import deque

class RateLimiter:
    """Thread-safe Sliding Window Rate Limiter"""
    
    def __init__(self):
        # Different buckets for different endpoints
        # Upstox General: ~10 calls/sec
        # Upstox Orders: ~10 calls/sec (conservative)
        self.limits = {
            'default': {'calls': 10, 'period': 1.0},
            'order': {'calls': 5, 'period': 1.0},
            'historical': {'calls': 3, 'period': 1.0}
        }
        self.records: Dict[str, Deque[float]] = {
            k: deque() for k in self.limits.keys()
        }
        self.lock = threading.Lock()
    
    def limit(self, bucket: str = 'default'):
        """Decorator to rate limit a function"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                self._wait_for_token(bucket)
                return func(*args, **kwargs)
            return wrapper
        return decorator

    def _wait_for_token(self, bucket: str):
        """Block until a token is available"""
        if bucket not in self.limits:
            bucket = 'default'
            
        limit_cfg = self.limits[bucket]
        max_calls = limit_cfg['calls']
        period = limit_cfg['period']
        
        with self.lock:
            history = self.records[bucket]
            
            while True:
                now = time.time()
                
                # Remove expired timestamps
                while history and now - history[0] > period:
                    history.popleft()
                
                if len(history) < max_calls:
                    history.append(now)
                    return
                
                # Wait until the oldest call expires
                sleep_time = period - (now - history[0])
                if sleep_time > 0:
                    time.sleep(sleep_time + 0.01) # Small buffer

# Global instance
limiter = RateLimiter()
