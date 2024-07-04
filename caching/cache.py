import os
import json
import redis
import diskcache
from typing import Any, Dict, Optional
from ..base.base import Cache


class Caching(Cache):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.cache_dir = config.get("cache", ".cache")
        self.seed = config.get("cache_seed", "1")
        self.cache = self.init_cache()

    def init_cache(self) -> Any:
        redis_url = self.config.get("redis_url")
        if redis_url:
            redis_cache = redis.Redis.from_url(redis_url)
            try:
                redis_cache.ping()
            except redis.RedisError as err:
                print(f"Failed to connect to Redis: {err}")
                return None
            return redis_cache, f"cache:{self.seed}:"
        return diskcache.Cache(os.path.join(self.cache_dir, str(self.seed)))

    def _prefixed_key(self, key: str) -> str:
        return f"rediscache:{self.seed}:{key}"

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        try:
            cache = self.cache
            if cache is None:
                return default
            if isinstance(cache, tuple) and cache[1]:
                result = cache[0].get(cache[1] + key)
            else:
                result = cache.get(key)
            return json.loads(result) if result is not None else default
        except Exception as err:
            print(f"Error getting value from cache: {err}")
            return default

    def set(self, key: str, value: Any) -> None:
        try:
            cache = self.cache
            if cache is None:
                return
            serialized_value = json.dumps(value)
            if isinstance(cache, tuple) and cache[1]:
                cache[0].set(cache[1] + key, serialized_value)
            else:
                cache.set(key, serialized_value)
        except Exception as err:
            print(f"Error setting value in cache: {err}")

    def close(self) -> None:
        try:
            cache = self.cache
            if cache is not None:
                if isinstance(cache, tuple) and cache[1]:
                    cache[0].close()
                else:
                    cache.close()
        except Exception as e:
            print(f"Error closing cache: {e}")

    def __enter__(self) -> "Caching":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
