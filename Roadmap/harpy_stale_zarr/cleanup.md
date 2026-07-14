# Remove Partial Zarr Elements After Failed Incremental Writes

## Status

Investigated and reproduced against Harpy 0.4.4. No Harpy fix has been
implemented yet.

## Scope

This document specifies the Harpy-side cleanup fix for failed incremental
writes to backed `SpatialData` stores. The cleanup must work even when a write
created a physical Zarr group that is too incomplete for SpatialData to
recognize as an element.

The napari-harpy caller bug that exposed this failure is separate: adding a row
to a Shapes element with an integer index currently combines the stored integer
identity with a generated string identity such as `__annotation_0`. That mixed
index fails Parquet serialization and must be fixed in napari-harpy. Fixing the
caller prevents this particular write error, but it does not make Harpy's
general failed-write cleanup safe.

## Reproduced Failure

Harpy's incremental overwrite uses a temporary element name:

```text
regions
    -> regions_<uuid>
```

The relevant early-write path in `harpy.utils._io._incremental_io_on_disk(...)`
is:

```python
temporary_element_name = f"{element_name}_{uuid.uuid4()}"
sdata[temporary_element_name] = element
try:
    sdata.write_element(temporary_element_name)
except Exception:
    if sdata.get(temporary_element_name) is not None:
        del sdata[temporary_element_name]
    sdata.delete_element_from_disk(temporary_element_name)
    raise
```

When serialization fails after Zarr has created the temporary group but before
it has written valid SpatialData metadata, the on-disk state is:

```text
shapes/regions_<uuid>/zarr.json

{
    "attributes": {},
    "zarr_format": 3,
    "node_type": "group"
}
```

There may be no `shapes.parquet`, `spatialdata_attrs.version`, or other metadata
that would let SpatialData classify the directory as a Shapes element.

The cleanup then fails for two related reasons:

1. Harpy removes the temporary element from the in-memory `SpatialData` object
   before asking SpatialData to delete it from disk.
2. `SpatialData.delete_element_from_disk(...)` discovers on-disk elements by
   enumerating valid SpatialData metadata. The incomplete group is therefore
   not recognized as an on-disk element and cannot be deleted through that API.

The original serialization exception is correctly re-raised, but the partial
group remains under the public `shapes/` container. Once consolidated metadata
includes that group, a later `read_zarr(...)` attempts to read it and fails at:

```python
version = _parse_version(...)
assert version is not None
```

This turns one failed element write into an unreadable SpatialData store.

## Required Cleanup Contract

Harpy's write transaction must satisfy these invariants:

1. A failed staging write must leave the original target element untouched.
2. Cleanup must address the exact physical group created for the staging name;
   it must not depend on SpatialData being able to parse that group.
3. Cleanup must remove the exact group only. It must never scan for and delete
   every name that merely shares a prefix with the target.
4. If the store uses consolidated metadata, cleanup must remove the deleted
   group from that metadata before the original write error is re-raised.
5. Cleanup errors must be logged, but they must never replace or mask the
   original serialization error.
6. The implementation must use the Zarr store abstraction rather than
   `shutil.rmtree(...)`, so local, remote, and non-filesystem stores retain the
   same behavior.
7. A successfully written staging element must not be deleted after the
   original target has already been removed unless the target has first been
   restored. At that point the staging element is the recoverable copy.

## Proposed Focused Implementation

### 1. Add physical-group cleanup independent of SpatialData parsing

Introduce one internal Harpy helper with an explicit element type and name:

```python
def _delete_element_group_from_store(
    sdata: SpatialData,
    *,
    element_type: Literal["images", "labels", "shapes", "tables", "points"],
    element_name: str,
) -> None:
    """Delete one exact physical element group and refresh metadata."""
```

The helper should:

1. Resolve the same store represented by `sdata.path` using a Zarr-compatible
   store-resolution helper.
2. Open the root group in write mode with `use_consolidated=False`; cleanup
   cannot trust possibly stale consolidated metadata.
3. Look up the exact `root[element_type][element_name]` child.
4. Delete that child when present and otherwise succeed idempotently.
5. Close the store even when deletion fails.
6. Rebuild consolidated metadata when the SpatialData store uses it.

The Zarr operation is conceptually:

```python
root = zarr.open_group(store=store, mode="r+", use_consolidated=False)
container = root[element_type]
if element_name in container:
    del container[element_name]
```

Store resolution should be isolated behind the helper. Harpy should not use a
local-path-only filesystem deletion as a shortcut.

### 2. Pass `element_type` into every cleanup boundary

`_incremental_io_on_disk(...)` already receives `element_type` and can pass it
directly to the physical cleanup helper.

`_write_element_with_cleanup(...)` currently receives only `element_name`.
Change its contract to receive `element_type` as well, and update the image,
labels, shapes, points, and table managers accordingly. Cleanup should not need
to infer an element's type from metadata that may be incomplete.

### 3. Clean a failed staging write by exact physical name

The staging-write exception path should detach the temporary in-memory element
and independently remove its physical group:

```python
try:
    sdata.write_element(temporary_element_name)
except Exception as write_error:
    if sdata.get(temporary_element_name) is not None:
        del sdata[temporary_element_name]
    try:
        _delete_element_group_from_store(
            sdata,
            element_type=element_type,
            element_name=temporary_element_name,
        )
    except Exception as cleanup_error:
        log.warning(
            "Physical cleanup of the failed temporary element also failed: "
            f"{cleanup_error}"
        )
    raise
```

The bare `raise` is important: callers must receive the original serialization
exception, not a secondary cleanup exception.

### 4. Make later transaction cleanup phase-aware

Wrap the rest of `_incremental_io_on_disk(...)` in a transaction boundary that
tracks whether the original target has been deleted.

```text
staging write fails
    -> delete partial staging group directly
    -> original target remains
    -> re-raise original write error

staging write succeeds
    -> later failure before target deletion
    -> delete valid staging group directly
    -> original target remains
    -> re-raise original error

target has been deleted
    -> replacement write fails
    -> remove any partial replacement target
    -> retain the valid staging element as the recovery copy
    -> report its exact name
    -> re-raise original error

replacement succeeds
    -> remove the staging element
    -> refresh consolidated metadata
```

This phase distinction prevents cleanup from deleting the only valid copy after
the overwrite has entered its destructive phase. Automatically restoring the
target from staging can be considered separately; it is not required for the
focused stale-group fix.

## Regression Coverage

### Failed staging serialization

Create a backed SpatialData store with an existing Shapes element, then inject
a failure after the temporary Zarr group is created but before valid element
metadata is written.

Assert that:

- the original exception propagates;
- the original target still exists and is unchanged;
- no UUID staging child remains in `root["shapes"]`;
- consolidated metadata contains no staging path;
- a fresh `read_zarr(...)` succeeds.

The injected failure must reproduce a physically created but semantically
invalid group. Raising before `sdata.write_element(...)` starts would not cover
the cleanup bug.

### Failure after a valid staging write but before target deletion

Inject a failure in the read/materialization step immediately after the staging
element has been written successfully.

Assert that:

- the original target remains unchanged;
- the valid staging element is removed;
- no staging path remains in consolidated metadata;
- the store is readable.

### Failure after target deletion

Inject a replacement-write failure after the original target has been deleted.

Assert that:

- any invalid partial target group is removed;
- the valid staging element is retained;
- the original replacement exception propagates;
- the warning reports the exact recovery element name;
- the store remains readable through the valid staging element.

### Cleanup failure preserves the primary error

Inject both a serialization failure and a physical cleanup failure.

Assert that the serialization exception remains the raised exception and that
the cleanup failure is present only in the warning/log context.

## Acceptance Criteria

- A failed temporary write cannot leave an unreadable element group behind.
- `read_zarr(...)` continues to work after an early incremental-write failure.
- The existing target is unchanged when staging fails.
- Consolidated and unconsolidated views of the store agree after cleanup.
- Cleanup is exact-name, idempotent, and store-backend agnostic.
- The original write exception is always preserved.
- A valid staging copy is retained once replacement has entered its destructive
  phase, unless the target is restored first.

## Out of Scope

- Choosing compatible identities for newly added rows in integer-indexed
  Shapes elements; that belongs in napari-harpy.
- Making the complete overwrite operation crash-atomic across process death or
  machine failure.
- Silently swallowing serialization or replacement errors.
- Broadly deleting UUID-looking elements without knowing that the current
  transaction created them.
