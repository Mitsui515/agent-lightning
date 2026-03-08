# Copyright (c) Microsoft. All rights reserved.

from .base import LightningStore, LightningStoreCapabilities, LightningStoreStatistics
from .client_server import LightningStoreClient, LightningStoreServer
from .collection_based import CollectionBasedLightningStore
from .memory import InMemoryLightningStore
from .threading import LightningStoreThreaded

__all__ = [
    "LightningStore",
    "LightningStoreCapabilities",
    "LightningStoreStatistics",
    "LightningStoreClient",
    "LightningStoreServer",
    "InMemoryLightningStore",
    "CollectionBasedLightningStore",
    "LightningStoreThreaded",
]


def _lazy_sqlite() -> type:
    """Lazily import SQLiteLightningStore to avoid ImportError when aiosqlite is not installed."""
    from .sqlite import SQLiteLightningStore

    return SQLiteLightningStore
