"""Diskcache wrapper for caching API calls keyed by arbitrary hashable tuples."""

import hashlib
import json
from typing import Any, Callable, Optional

import diskcache

from prefvlm.config import cfg

_cache: Optional[diskcache.Cache] = None


def get_cache() -> diskcache.Cache:
    global _cache
    if _cache is None:
        _cache = diskcache.Cache(str(cfg.paths.cache_dir), size_limit=2**33)  # 8 GB
    return _cache


def _make_key(namespace: str, *parts: Any) -> str:
    """Stable string key from namespace + arbitrary parts."""
    raw = json.dumps([namespace, *parts], sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"{namespace}:{digest}"


def cached(namespace: str, *key_parts: Any) -> Optional[Any]:
    """Return cached value or None if not found."""
    return get_cache().get(_make_key(namespace, *key_parts))


def store(namespace: str, value: Any, *key_parts: Any) -> None:
    """Store value under the derived key."""
    get_cache().set(_make_key(namespace, *key_parts), value)


def cached_call(
    namespace: str,
    key_parts: tuple,
    fn: Callable[[], Any],
) -> Any:
    """Return cached result or call fn(), store, and return."""
    key = _make_key(namespace, *key_parts)
    cache = get_cache()
    hit = cache.get(key)
    if hit is not None:
        return hit
    result = fn()
    cache.set(key, result)
    return result


def clear_namespace(namespace: str) -> int:
    """Delete all keys in a namespace. Returns count deleted."""
    cache = get_cache()
    keys_to_delete = [k for k in cache if isinstance(k, str) and k.startswith(f"{namespace}:")]
    for k in keys_to_delete:
        del cache[k]
    return len(keys_to_delete)
