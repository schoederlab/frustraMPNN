"""
Caching utilities for FrustraMPNN training.

This module provides a disk-based caching decorator for expensive operations
like PDB parsing. The cache is keyed by a user-defined function and stored
in the platform's cache directory.

Original source: test_data/training/cache.py
"""

from __future__ import annotations

import functools
import inspect
import os
import pickle
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:

    pass

F = TypeVar("F", bound=Callable[..., Any])


def stringify_cache_key(key: Any) -> str:
    """
    Convert a cache key to a unique string identifier.

    Uses UUID3 with DNS namespace to generate a deterministic hash
    from the string representation of the key.

    Args:
        key: Any hashable object to convert to string

    Returns:
        Hexadecimal string identifier
    """
    return uuid.uuid3(uuid.NAMESPACE_DNS, str(key)).hex


def cache(
    cache_key: Callable[..., Any],
    version: float = 0.0,
    disable: bool = False,
) -> Callable[[F], F]:
    """
    Cache the result of a function call on disk for speedup.

    This decorator caches function results to disk based on a user-defined
    cache key function. The cache is stored in the platform's cache directory
    under functions/{function_name}/{version}/{key}.pkl.

    Args:
        cache_key: Function that takes the same arguments as the decorated
            function and returns a hashable cache key
        version: Cache version number (increment to invalidate old caches)
        disable: If True, bypass the cache entirely

    Returns:
        Decorated function with caching

    Example:
        >>> @cache(lambda cfg, pdb_file: pdb_file)
        ... def parse_pdb_cached(cfg, pdb_file):
        ...     return parse_PDB(pdb_file)
    """

    def inner_cache(f: F) -> F:
        f_sig = inspect.signature(f)

        @functools.wraps(f)
        def cached_f(cfg: Any, *args: Any, **kwargs: Any) -> Any:
            # Ensure that default args are properly passed to cache key
            bound = f_sig.bind(cfg, *args, **kwargs)
            bound.apply_defaults()
            _, *bound_args = bound.args
            bound_kwargs = bound.kwargs

            # Generate cache key
            key = stringify_cache_key(cache_key(cfg, *bound_args, **bound_kwargs))

            # Get cache directory from config
            cache_dir = getattr(getattr(cfg, "platform", cfg), "cache_dir", "cache")
            cache_file = f"{cache_dir}/functions/{f.__name__}/{version}/{key}.pkl"

            if not disable:
                try:
                    with open(cache_file, "rb") as fh:
                        ret = pickle.load(fh)
                        return ret
                except (FileNotFoundError, EOFError):
                    pass

            # Call the actual function
            ret = f(cfg, *args, **kwargs)

            # Save to cache
            cache_folder = "/".join(cache_file.split("/")[:-1])
            os.makedirs(cache_folder, exist_ok=True)
            with open(cache_file, "wb") as fh:
                pickle.dump(ret, fh)

            return ret

        return cached_f  # type: ignore

    return inner_cache


# Pre-configured cached PDB parser
def get_cached_pdb_parser(
    cfg: Any,
) -> Callable[[str], Any]:
    """
    Get a cached PDB parser function.

    This returns a function that parses PDB files with caching enabled.
    The cache is stored in the platform's cache directory.

    Args:
        cfg: Configuration object with platform.cache_dir

    Returns:
        Cached PDB parsing function

    Example:
        >>> parse_pdb = get_cached_pdb_parser(cfg)
        >>> pdb = parse_pdb(pdb_file)
    """
    from frustrampnn.model.pdb_parsing import parse_PDB

    @cache(lambda cfg, pdb_file: pdb_file)
    def parse_pdb_cached(cfg: Any, pdb_file: str) -> Any:
        return parse_PDB(pdb_file)

    return lambda pdb_file: parse_pdb_cached(cfg, pdb_file)

