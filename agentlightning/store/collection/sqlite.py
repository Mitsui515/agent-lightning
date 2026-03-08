# Copyright (c) Microsoft. All rights reserved.

"""SQLite-backed collection implementations for LightningStore.

All collections share a single ``aiosqlite`` connection managed by
:class:`SQLiteLightningCollections`. An :class:`asyncio.Lock` serialises
concurrent coroutines; SQLite transactions are used to ensure atomicity.

Each Pydantic model is serialised to a single JSON column (``_data``).
Primary-key columns are stored separately so that ``INSERT OR REPLACE``
and ``SELECT … WHERE pk = ?`` work without unpacking JSON.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)

try:
    import aiosqlite
except ImportError as _exc:
    raise ImportError(
        "The SQLite backend requires the 'aiosqlite' package. "
        "Install it with: pip install 'agentlightning[sqlite]'"
    ) from _exc

from pydantic import BaseModel, TypeAdapter

from agentlightning.types import (
    Attempt,
    FilterOptions,
    PaginatedResult,
    ResourcesUpdate,
    Rollout,
    SortOptions,
    Span,
    Worker,
)
from agentlightning.utils.metrics import MetricsBackend

from .base import (
    AtomicLabels,
    AtomicMode,
    Collection,
    DuplicatedPrimaryKeyError,
    KeyValue,
    LightningCollections,
    Queue,
    get_sort_value,
    item_matches_filters,
    ensure_numeric,
    normalize_filter_options,
    resolve_sort_options,
    tracked,
)

if TYPE_CHECKING:
    pass

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize_identifier(name: str) -> str:
    """Return a safe SQLite identifier (alphanumeric + underscore only)."""
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    if not sanitized or sanitized[0].isdigit():
        sanitized = "t_" + sanitized
    return sanitized


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


class SQLiteBasedCollection(Collection[T], Generic[T]):
    """SQLite-backed ``Collection`` implementation.

    Each item is stored as a JSON blob in the ``_data`` column. Primary-key
    field values are duplicated into dedicated columns for efficient filtering.

    Args:
        conn: The shared ``aiosqlite`` connection managed by
            :class:`SQLiteLightningCollections`.
        table_name: SQLite table name for this collection.
        item_type: Pydantic model class stored in the collection.
        primary_keys: Ordered sequence of primary-key field names.
        tracker: Optional metrics backend.
    """

    def __init__(
        self,
        conn: aiosqlite.Connection,
        table_name: str,
        item_type: Type[T],
        primary_keys: Sequence[str],
        tracker: Optional[MetricsBackend] = None,
    ) -> None:
        super().__init__(tracker=tracker)
        self._conn = conn
        self._table = _sanitize_identifier(table_name)
        self._item_type = item_type
        self._primary_keys: Tuple[str, ...] = tuple(primary_keys)
        self._adapter: TypeAdapter[T] = TypeAdapter(item_type)  # type: ignore[arg-type]

    @property
    def collection_name(self) -> str:
        return self._table

    def primary_keys(self) -> Sequence[str]:
        return self._primary_keys

    def item_type(self) -> Type[T]:
        return self._item_type

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    async def ensure_table(self) -> None:
        pk_cols = ", ".join(f"{_sanitize_identifier(pk)} TEXT NOT NULL" for pk in self._primary_keys)
        pk_constraint = ", ".join(_sanitize_identifier(pk) for pk in self._primary_keys)
        await self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                {pk_cols},
                _data TEXT NOT NULL,
                _insert_order INTEGER PRIMARY KEY AUTOINCREMENT,
                UNIQUE ({pk_constraint})
            )
            """
        )
        # Index each primary key column for faster lookups.
        for pk in self._primary_keys:
            safe_pk = _sanitize_identifier(pk)
            await self._conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self._table}_{safe_pk} ON {self._table} ({safe_pk})"
            )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def _serialize(self, item: T) -> Tuple[Tuple[Any, ...], str]:
        """Return ``(pk_values, json_str)`` for *item*."""
        pk_values = tuple(getattr(item, pk) for pk in self._primary_keys)
        if isinstance(item, BaseModel):
            data_json = item.model_dump_json()
        else:
            data_json = json.dumps(item)
        return pk_values, data_json

    def _deserialize(self, data_json: str) -> T:
        """Deserialise a JSON string back to *T*."""
        return self._adapter.validate_json(data_json)

    # ------------------------------------------------------------------
    # Size
    # ------------------------------------------------------------------

    @tracked("size")
    async def size(self) -> int:
        async with self._conn.execute(f"SELECT COUNT(*) FROM {self._table}") as cursor:
            row = await cursor.fetchone()
        return int(row[0]) if row else 0  # type: ignore[index]

    # ------------------------------------------------------------------
    # Query / get
    # ------------------------------------------------------------------

    def _build_pk_where(
        self, filters_map: Optional[Mapping[str, Any]]
    ) -> Tuple[str, List[Any]]:
        """Build a ``WHERE`` clause for exact primary-key matches.

        Returns ``(where_clause, params)``.  If no exact primary-key
        constraint exists, the clause is empty.
        """
        if not filters_map:
            return "", []
        clauses: List[str] = []
        params: List[Any] = []
        for pk in self._primary_keys:
            ops = filters_map.get(pk)
            if ops and "exact" in ops and ops["exact"] is not None:
                clauses.append(f"{_sanitize_identifier(pk)} = ?")
                params.append(str(ops["exact"]))
            else:
                break  # stop at first missing exact constraint
        if not clauses:
            return "", []
        return "WHERE " + " AND ".join(clauses), params

    @tracked("query")
    async def query(
        self,
        filter: Optional[FilterOptions] = None,
        sort: Optional[SortOptions] = None,
        limit: int = -1,
        offset: int = 0,
    ) -> PaginatedResult[T]:
        filters_map, must_filters, filter_logic = normalize_filter_options(filter)
        sort_by, sort_order = resolve_sort_options(sort)

        where_clause, params = self._build_pk_where(filters_map)
        sql = f"SELECT _data FROM {self._table} {where_clause} ORDER BY _insert_order ASC"

        async with self._conn.execute(sql, params) as cursor:
            rows = await cursor.fetchall()

        items: List[T] = []
        for (data_json,) in rows:
            item = self._deserialize(data_json)
            if item_matches_filters(item, filters_map, filter_logic, must_filters):
                items.append(item)

        total = len(items)

        if sort_by:
            reverse = sort_order == "desc"
            items.sort(key=lambda x: get_sort_value(x, sort_by), reverse=reverse)

        if limit == -1:
            paged = items[offset:]
        else:
            paged = items[offset : offset + limit]

        return PaginatedResult(items=paged, limit=limit, offset=offset, total=total)

    @tracked("get")
    async def get(
        self,
        filter: Optional[FilterOptions] = None,
        sort: Optional[SortOptions] = None,
    ) -> Optional[T]:
        filters_map, must_filters, filter_logic = normalize_filter_options(filter)
        sort_by, sort_order = resolve_sort_options(sort)

        where_clause, params = self._build_pk_where(filters_map)
        sql = f"SELECT _data FROM {self._table} {where_clause} ORDER BY _insert_order ASC"

        async with self._conn.execute(sql, params) as cursor:
            rows = await cursor.fetchall()

        candidates: List[T] = []
        for (data_json,) in rows:
            item = self._deserialize(data_json)
            if item_matches_filters(item, filters_map, filter_logic, must_filters):
                candidates.append(item)

        if not candidates:
            return None
        if not sort_by:
            return candidates[0]

        reverse = sort_order == "desc"
        candidates.sort(key=lambda x: get_sort_value(x, sort_by), reverse=reverse)
        return candidates[0]

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    @tracked("insert")
    async def insert(self, items: Sequence[T]) -> None:
        if not items:
            return
        pk_col_names = ", ".join(_sanitize_identifier(pk) for pk in self._primary_keys)
        placeholders = ", ".join("?" for _ in self._primary_keys)
        sql = f"INSERT INTO {self._table} ({pk_col_names}, _data) VALUES ({placeholders}, ?)"

        seen: set[Tuple[Any, ...]] = set()
        for item in items:
            pk_values, data_json = self._serialize(item)
            if pk_values in seen:
                raise DuplicatedPrimaryKeyError(
                    f"Duplicate primary key in batch: {dict(zip(self._primary_keys, pk_values))}"
                )
            seen.add(pk_values)
            try:
                await self._conn.execute(sql, (*pk_values, data_json))
            except Exception as exc:
                # aiosqlite re-raises sqlite3.IntegrityError for UNIQUE violations.
                import sqlite3

                if isinstance(exc.__cause__, sqlite3.IntegrityError) or isinstance(exc, sqlite3.IntegrityError):
                    raise DuplicatedPrimaryKeyError(
                        f"Item already exists: {dict(zip(self._primary_keys, pk_values))}"
                    ) from exc
                raise

    @tracked("update")
    async def update(self, items: Sequence[T], update_fields: Sequence[str] | None = None) -> Sequence[T]:
        updated: List[T] = []
        for item in items:
            pk_values, _ = self._serialize(item)
            # Check that the item exists first.
            where = " AND ".join(f"{_sanitize_identifier(pk)} = ?" for pk in self._primary_keys)
            async with self._conn.execute(
                f"SELECT _data FROM {self._table} WHERE {where}", list(pk_values)
            ) as cursor:
                existing_row = await cursor.fetchone()

            if existing_row is None:
                raise ValueError(
                    f"Item does not exist: {dict(zip(self._primary_keys, pk_values))}"
                )

            if update_fields is not None and isinstance(item, BaseModel) and existing_row:
                existing_item = self._deserialize(existing_row[0])
                if isinstance(existing_item, BaseModel):
                    merged = existing_item.model_copy(
                        update={f: getattr(item, f) for f in update_fields}
                    )
                    _, data_json = self._serialize(merged)  # type: ignore[arg-type]
                    new_item: T = merged  # type: ignore[assignment]
                else:
                    _, data_json = self._serialize(item)
                    new_item = item
            else:
                _, data_json = self._serialize(item)
                new_item = item

            await self._conn.execute(
                f"UPDATE {self._table} SET _data = ? WHERE {where}",
                (data_json, *pk_values),
            )
            updated.append(new_item)
        return updated

    @tracked("upsert")
    async def upsert(self, items: Sequence[T], update_fields: Sequence[str] | None = None) -> Sequence[T]:
        upserted: List[T] = []
        for item in items:
            pk_values, data_json = self._serialize(item)
            where = " AND ".join(f"{_sanitize_identifier(pk)} = ?" for pk in self._primary_keys)
            async with self._conn.execute(
                f"SELECT _data FROM {self._table} WHERE {where}", list(pk_values)
            ) as cursor:
                existing_row = await cursor.fetchone()

            if existing_row is None:
                # Insert
                pk_col_names = ", ".join(_sanitize_identifier(pk) for pk in self._primary_keys)
                placeholders = ", ".join("?" for _ in self._primary_keys)
                await self._conn.execute(
                    f"INSERT INTO {self._table} ({pk_col_names}, _data) VALUES ({placeholders}, ?)",
                    (*pk_values, data_json),
                )
                upserted.append(item)
            elif update_fields is None:
                # Replace all fields
                await self._conn.execute(
                    f"UPDATE {self._table} SET _data = ? WHERE {where}",
                    (data_json, *pk_values),
                )
                upserted.append(item)
            elif len(update_fields) == 0:
                # get_or_insert semantics: keep existing
                existing_item = self._deserialize(existing_row[0])
                upserted.append(existing_item)
            else:
                # Partial update
                existing_item = self._deserialize(existing_row[0])
                if isinstance(existing_item, BaseModel) and isinstance(item, BaseModel):
                    merged = existing_item.model_copy(
                        update={f: getattr(item, f) for f in update_fields}
                    )
                    _, merged_json = self._serialize(merged)  # type: ignore[arg-type]
                    await self._conn.execute(
                        f"UPDATE {self._table} SET _data = ? WHERE {where}",
                        (merged_json, *pk_values),
                    )
                    upserted.append(merged)  # type: ignore[arg-type]
                else:
                    await self._conn.execute(
                        f"UPDATE {self._table} SET _data = ? WHERE {where}",
                        (data_json, *pk_values),
                    )
                    upserted.append(item)
        return upserted

    @tracked("delete")
    async def delete(self, items: Sequence[T]) -> None:
        for item in items:
            pk_values, _ = self._serialize(item)
            where = " AND ".join(f"{_sanitize_identifier(pk)} = ?" for pk in self._primary_keys)
            async with self._conn.execute(
                f"SELECT _data FROM {self._table} WHERE {where}", list(pk_values)
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                raise ValueError(f"Item does not exist: {dict(zip(self._primary_keys, pk_values))}")
            await self._conn.execute(
                f"DELETE FROM {self._table} WHERE {where}", list(pk_values)
            )


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------


class SQLiteBasedQueue(Queue[T], Generic[T]):
    """SQLite-backed :class:`Queue` implementation.

    Items are stored in insertion order using an ``AUTOINCREMENT`` sequence
    column. Dequeue removes the oldest rows first (FIFO).

    Args:
        conn: Shared ``aiosqlite`` connection.
        table_name: SQLite table name.
        item_type: Python type of queued items (``str`` for rollout IDs).
        tracker: Optional metrics backend.
    """

    def __init__(
        self,
        conn: aiosqlite.Connection,
        table_name: str,
        item_type: Type[T],
        tracker: Optional[MetricsBackend] = None,
    ) -> None:
        super().__init__(tracker=tracker)
        self._conn = conn
        self._table = _sanitize_identifier(table_name)
        self._item_type = item_type

    @property
    def collection_name(self) -> str:
        return self._table

    def item_type(self) -> Type[T]:
        return self._item_type

    async def ensure_table(self) -> None:
        await self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                _seq INTEGER PRIMARY KEY AUTOINCREMENT,
                _data TEXT NOT NULL
            )
            """
        )

    def _serialize(self, item: T) -> str:
        if isinstance(item, BaseModel):
            return item.model_dump_json()
        return json.dumps(item)

    def _deserialize(self, data_json: str) -> T:
        if issubclass(self._item_type, BaseModel):
            adapter: TypeAdapter[T] = TypeAdapter(self._item_type)  # type: ignore[arg-type]
            return adapter.validate_json(data_json)
        return json.loads(data_json)  # type: ignore[return-value]

    @tracked("has")
    async def has(self, item: T) -> bool:
        data_json = self._serialize(item)
        async with self._conn.execute(
            f"SELECT COUNT(*) FROM {self._table} WHERE _data = ?", (data_json,)
        ) as cursor:
            row = await cursor.fetchone()
        return bool(row and row[0] > 0)  # type: ignore[index]

    @tracked("enqueue")
    async def enqueue(self, items: Sequence[T]) -> Sequence[T]:
        for item in items:
            await self._conn.execute(
                f"INSERT INTO {self._table} (_data) VALUES (?)", (self._serialize(item),)
            )
        return items

    @tracked("dequeue")
    async def dequeue(self, limit: int = 1) -> Sequence[T]:
        if limit <= 0:
            return []
        async with self._conn.execute(
            f"SELECT _seq, _data FROM {self._table} ORDER BY _seq ASC LIMIT ?", (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
        if not rows:
            return []
        seq_ids = [row[0] for row in rows]
        placeholders = ", ".join("?" for _ in seq_ids)
        await self._conn.execute(
            f"DELETE FROM {self._table} WHERE _seq IN ({placeholders})", seq_ids
        )
        return [self._deserialize(row[1]) for row in rows]

    @tracked("peek")
    async def peek(self, limit: int = 1) -> Sequence[T]:
        if limit <= 0:
            return []
        async with self._conn.execute(
            f"SELECT _data FROM {self._table} ORDER BY _seq ASC LIMIT ?", (limit,)
        ) as cursor:
            rows = await cursor.fetchall()
        return [self._deserialize(row[0]) for row in rows]

    @tracked("size")
    async def size(self) -> int:
        async with self._conn.execute(f"SELECT COUNT(*) FROM {self._table}") as cursor:
            row = await cursor.fetchone()
        return int(row[0]) if row else 0  # type: ignore[index]


# ---------------------------------------------------------------------------
# KeyValue
# ---------------------------------------------------------------------------


class SQLiteBasedKeyValue(KeyValue[K, V], Generic[K, V]):
    """SQLite-backed :class:`KeyValue` implementation.

    Keys are stored as JSON strings; values are stored as JSON strings.

    Args:
        conn: Shared ``aiosqlite`` connection.
        table_name: SQLite table name.
        key_type: Python type for keys.
        value_type: Python type for values.
        tracker: Optional metrics backend.
    """

    def __init__(
        self,
        conn: aiosqlite.Connection,
        table_name: str,
        key_type: Type[K],
        value_type: Type[V],
        tracker: Optional[MetricsBackend] = None,
    ) -> None:
        super().__init__(tracker=tracker)
        self._conn = conn
        self._table = _sanitize_identifier(table_name)
        self._key_type = key_type
        self._value_type = value_type

    @property
    def collection_name(self) -> str:
        return self._table

    async def ensure_table(self) -> None:
        await self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                _key TEXT PRIMARY KEY,
                _value TEXT NOT NULL
            )
            """
        )

    def _serialize_key(self, key: K) -> str:
        return json.dumps(key)

    def _serialize_value(self, value: V) -> str:
        if isinstance(value, BaseModel):
            return value.model_dump_json()
        return json.dumps(value)

    def _deserialize_value(self, raw: str) -> V:
        if issubclass(self._value_type, BaseModel):
            adapter: TypeAdapter[V] = TypeAdapter(self._value_type)  # type: ignore[arg-type]
            return adapter.validate_json(raw)
        return json.loads(raw)  # type: ignore[return-value]

    @tracked("has")
    async def has(self, key: K) -> bool:
        async with self._conn.execute(
            f"SELECT COUNT(*) FROM {self._table} WHERE _key = ?", (self._serialize_key(key),)
        ) as cursor:
            row = await cursor.fetchone()
        return bool(row and row[0] > 0)  # type: ignore[index]

    @tracked("get")
    async def get(self, key: K, default: V | None = None) -> V | None:
        async with self._conn.execute(
            f"SELECT _value FROM {self._table} WHERE _key = ?", (self._serialize_key(key),)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return default
        return self._deserialize_value(row[0])  # type: ignore[index]

    @tracked("set")
    async def set(self, key: K, value: V) -> None:
        await self._conn.execute(
            f"INSERT INTO {self._table} (_key, _value) VALUES (?, ?) "
            f"ON CONFLICT(_key) DO UPDATE SET _value = excluded._value",
            (self._serialize_key(key), self._serialize_value(value)),
        )

    @tracked("inc")
    async def inc(self, key: K, amount: V) -> V:
        assert ensure_numeric(amount, description="amount")
        serialized_key = self._serialize_key(key)
        async with self._conn.execute(
            f"SELECT _value FROM {self._table} WHERE _key = ?", (serialized_key,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is not None:
            current: V = self._deserialize_value(row[0])  # type: ignore[index]
            assert ensure_numeric(current, description=f"value for key {key!r}")
            new_value: V = current + amount  # type: ignore[operator]
        else:
            new_value = amount
        await self._conn.execute(
            f"INSERT INTO {self._table} (_key, _value) VALUES (?, ?) "
            f"ON CONFLICT(_key) DO UPDATE SET _value = excluded._value",
            (serialized_key, self._serialize_value(new_value)),
        )
        return new_value

    @tracked("chmax")
    async def chmax(self, key: K, value: V) -> V:
        assert ensure_numeric(value, description="value")
        serialized_key = self._serialize_key(key)
        async with self._conn.execute(
            f"SELECT _value FROM {self._table} WHERE _key = ?", (serialized_key,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is not None:
            current = self._deserialize_value(row[0])  # type: ignore[index]
            assert ensure_numeric(current, description=f"value for key {key!r}")
            if value > current:  # type: ignore[operator]
                await self._conn.execute(
                    f"UPDATE {self._table} SET _value = ? WHERE _key = ?",
                    (self._serialize_value(value), serialized_key),
                )
                return value
            return current  # type: ignore[return-value]
        else:
            await self._conn.execute(
                f"INSERT INTO {self._table} (_key, _value) VALUES (?, ?)",
                (serialized_key, self._serialize_value(value)),
            )
            return value

    @tracked("pop")
    async def pop(self, key: K, default: V | None = None) -> V | None:
        serialized_key = self._serialize_key(key)
        async with self._conn.execute(
            f"SELECT _value FROM {self._table} WHERE _key = ?", (serialized_key,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return default
        await self._conn.execute(
            f"DELETE FROM {self._table} WHERE _key = ?", (serialized_key,)
        )
        return self._deserialize_value(row[0])  # type: ignore[index]

    @tracked("size")
    async def size(self) -> int:
        async with self._conn.execute(f"SELECT COUNT(*) FROM {self._table}") as cursor:
            row = await cursor.fetchone()
        return int(row[0]) if row else 0  # type: ignore[index]


# ---------------------------------------------------------------------------
# LightningCollections
# ---------------------------------------------------------------------------


class SQLiteLightningCollections(LightningCollections):
    """SQLite-based implementation of :class:`LightningCollections`.

    All collections share one ``aiosqlite`` connection to a single SQLite
    database file (or ``:memory:``). A single :class:`asyncio.Lock` serialises
    concurrent write access; SQLite transactions provide atomicity.

    Args:
        db_path: Path to the SQLite database file, or ``":memory:"`` for an
            in-memory database.
        tracker: Optional metrics backend.
    """

    def __init__(
        self,
        db_path: str,
        tracker: Optional[MetricsBackend] = None,
    ) -> None:
        super().__init__(tracker=tracker)
        self._db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None
        self._lock: asyncio.Lock = asyncio.Lock()

        # These are set up after _ensure_initialized() is called.
        self._rollouts: Optional[SQLiteBasedCollection[Rollout]] = None
        self._attempts: Optional[SQLiteBasedCollection[Attempt]] = None
        self._spans: Optional[SQLiteBasedCollection[Span]] = None
        self._resources: Optional[SQLiteBasedCollection[ResourcesUpdate]] = None
        self._workers: Optional[SQLiteBasedCollection[Worker]] = None
        self._rollout_queue: Optional[SQLiteBasedQueue[str]] = None
        self._span_sequence_ids: Optional[SQLiteBasedKeyValue[str, int]] = None

    @property
    def collection_name(self) -> str:
        return "sqlite_collections"

    async def initialize(self) -> None:
        """Open the SQLite connection and create all required tables.

        Must be called once before using any collection.
        """
        if self._conn is not None:
            return
        # isolation_level=None disables pysqlite's implicit transaction management
        # so we can issue explicit BEGIN/COMMIT/ROLLBACK statements ourselves.
        self._conn = await aiosqlite.connect(self._db_path, isolation_level=None)
        # WAL (Write-Ahead Logging) allows concurrent readers while a writer is
        # active, which greatly reduces lock contention for our usage pattern
        # (many reads, occasional writes).
        await self._conn.execute("PRAGMA journal_mode=WAL")
        # NORMAL durability is a safe trade-off: data is written to disk before
        # returning to the caller but without a full fsync on every transaction.
        # This avoids data loss in most failure scenarios while remaining faster
        # than the FULL setting.
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")

        # Build collection instances.
        self._rollouts = SQLiteBasedCollection(
            self._conn, "rollouts", Rollout, ["rollout_id"], tracker=self._tracker
        )
        self._attempts = SQLiteBasedCollection(
            self._conn, "attempts", Attempt, ["rollout_id", "attempt_id"], tracker=self._tracker
        )
        self._spans = SQLiteBasedCollection(
            self._conn, "spans", Span, ["rollout_id", "attempt_id", "span_id"], tracker=self._tracker
        )
        self._resources = SQLiteBasedCollection(
            self._conn, "resources", ResourcesUpdate, ["resources_id"], tracker=self._tracker
        )
        self._workers = SQLiteBasedCollection(
            self._conn, "workers", Worker, ["worker_id"], tracker=self._tracker
        )
        self._rollout_queue = SQLiteBasedQueue(
            self._conn, "rollout_queue", str, tracker=self._tracker
        )
        self._span_sequence_ids = SQLiteBasedKeyValue(
            self._conn, "span_sequence_ids", str, int, tracker=self._tracker
        )

        # Create all tables.
        await self._conn.execute("BEGIN")
        try:
            await self._rollouts.ensure_table()
            await self._attempts.ensure_table()
            await self._spans.ensure_table()
            await self._resources.ensure_table()
            await self._workers.ensure_table()
            await self._rollout_queue.ensure_table()
            await self._span_sequence_ids.ensure_table()
            await self._conn.execute("COMMIT")
        except Exception:
            await self._conn.execute("ROLLBACK")
            raise

    async def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Collection accessors (raise if not initialised)
    # ------------------------------------------------------------------

    def _require_initialized(self) -> None:
        if self._conn is None:
            raise RuntimeError(
                "SQLiteLightningCollections has not been initialised. "
                "Call await collections.initialize() first."
            )

    @property
    def rollouts(self) -> SQLiteBasedCollection[Rollout]:
        self._require_initialized()
        assert self._rollouts is not None
        return self._rollouts

    @property
    def attempts(self) -> SQLiteBasedCollection[Attempt]:
        self._require_initialized()
        assert self._attempts is not None
        return self._attempts

    @property
    def spans(self) -> SQLiteBasedCollection[Span]:
        self._require_initialized()
        assert self._spans is not None
        return self._spans

    @property
    def resources(self) -> SQLiteBasedCollection[ResourcesUpdate]:
        self._require_initialized()
        assert self._resources is not None
        return self._resources

    @property
    def workers(self) -> SQLiteBasedCollection[Worker]:
        self._require_initialized()
        assert self._workers is not None
        return self._workers

    @property
    def rollout_queue(self) -> SQLiteBasedQueue[str]:
        self._require_initialized()
        assert self._rollout_queue is not None
        return self._rollout_queue

    @property
    def span_sequence_ids(self) -> SQLiteBasedKeyValue[str, int]:
        self._require_initialized()
        assert self._span_sequence_ids is not None
        return self._span_sequence_ids

    # ------------------------------------------------------------------
    # Atomic context manager
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def atomic(
        self,
        *,
        mode: AtomicMode = "rw",
        snapshot: bool = False,
        commit: bool = False,
        labels: Optional[Sequence[AtomicLabels]] = None,
        **kwargs: Any,
    ):
        """Serialise access with a lock and wrap writes in a SQLite transaction.

        For read-only operations (``mode="r"`` and ``snapshot=False``) the
        lock is **not** acquired; this allows concurrent reads.
        """
        self._require_initialized()
        assert self._conn is not None

        if mode == "r" and not snapshot:
            # Allow concurrent reads without locking.
            yield self
            return

        async with self._lock:
            async with self.tracking_context(operation="atomic", collection=self.collection_name):
                await self._conn.execute("BEGIN EXCLUSIVE")
                try:
                    yield self
                    await self._conn.execute("COMMIT")
                except Exception:
                    try:
                        await self._conn.execute("ROLLBACK")
                    except Exception:
                        pass
                    raise

    @tracked("evict_spans_for_rollout")
    async def evict_spans_for_rollout(self, rollout_id: str) -> None:
        """Delete all spans belonging to a rollout from the database.

        Unlike the in-memory backend, the SQLite backend physically removes
        the rows rather than discarding in-process memory.
        """
        self._require_initialized()
        assert self._conn is not None
        await self._conn.execute(
            "DELETE FROM spans WHERE rollout_id = ?", (rollout_id,)
        )
