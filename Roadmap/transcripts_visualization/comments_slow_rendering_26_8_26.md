The benchmark now identifies the cause precisely. For AAMP at a 100,000-point budget, the dominant problem is the one-VisPy-resource-per-tile renderer. Cold Zarr decoding is also expensive, and two quadratic residency bookkeeping paths add avoidable delay.

The benchmark used:

- Full-extent AAMP viewport
- Exact L0
- 60,512 returned points
- 4,453 logical tiles
- 69 physical buckets
- Real macOS OpenGL canvas, 1,000 × 800
- Application-cold caches; operating-system filesystem cache was not cleared

The full report is saved at:

```text
/private/tmp/napari-harpy-aamp-profile.json
```

## 1. One-time cache startup

| Operation | Time | Resident memory |
|---|---:|---:|
| Reader entry and compact catalog | 148 ms | Small |
| Open/inspect 108 bucket stores and project indexes | 3.25 s | Metadata only |
| Load all bucket lookup indexes | 4.85 s | 568.4 MiB |
| Load AAMP selected-value index | 89 ms | 161 KiB |
| Construct empty VisPy layer | 168 ms | — |

Bucket-index projection and loading take approximately 8.1 seconds altogether, but only once when the cache session starts. They do not run again on ordinary zoom or pan requests.

Loading the bucket lookup indexes raised process RSS from approximately 320 MiB to 906 MiB.

## 2. Cold viewport worker

| Operation | Time |
|---|---:|
| Level selection | 0.25 ms |
| Viewport/tile planning | 21.59 ms |
| Read all requested bucket payloads | 4.22 s |
| CPU snapshot assembly/residency | approximately 0.86 s |
| Complete cold worker snapshot | **5.10 s** |

Level selection is effectively irrelevant here. Manifest planning is also comparatively small.

### Zarr payload-read breakdown

The 69 buckets are processed sequentially by [\_read_manifest_requests()](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/core/multi_scale_cache_points_zarr/reader.py:1583).

| Zarr array | Calls | Time | Returned bytes |
|---|---:|---:|---:|
| `location` | 69 | 2.27 s | 0.462 MiB |
| `value_id` | 69 | 1.86 s | 0.231 MiB |
| Combined | 138 | **4.14 s** | **0.693 MiB** |

The remaining non-Zarr bucket work was small:

- Sparse interval resolution for 4,453 tiles: 44.6 ms
- Exact row-selector construction: 4.9 ms
- Bucket grouping, result splitting and ordering: approximately 30 ms

The Zarr calls in [read_display_payloads()](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/core/multi_scale_cache_points_zarr/storage/bucket_reader.py:254) are therefore the cold worker bottleneck.

### Why 0.69 MiB takes four seconds

The 60,512 selected AAMP rows touch:

- 4,291 independently decoded 4,096-row chunks
- 1,067 physical shards
- approximately 17.57 million decoded rows from each aligned array

That represents approximately:

- 134 MiB decoded from `location`
- 67 MiB decoded from `value_id`
- 201 MiB combined uncompressed chunk content

The row-level decode amplification is approximately 290×:

```text
60,512 selected rows
        ↓
4,291 touched inner chunks
        ↓
17,570,347 decoded rows
```

This is why merely returning 0.69 MiB still takes seconds. Fewer buckets or concurrent bucket reads can reduce serial dispatch, but they do not automatically eliminate the underlying chunk amplification.

## 3. Unexpected CPU residency cost

The approximately 0.86-second CPU snapshot remainder came almost entirely from retaining 4,453 tiny tiles:

| CPU assembly step | Time |
|---|---:|
| Construct residency keys | 9.2 ms |
| Copy point arrays | 2.5 ms |
| Construct immutable render tiles | 7.7 ms |
| Insert into CPU residency | **851.6 ms** |
| Final snapshot tuple | 0.5 ms |

The problem is [\_evict_until_fits()](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/viewer/tiled_points/runtime/residency.py:126). It materializes the complete, growing key collection for every inserted tile, even when no eviction is needed. With thousands of tiles, that becomes effectively quadratic.

This is a concrete implementation defect and can be fixed independently.

## 4. Qt delivery

Steady-state queued delivery of an AAMP-shaped immutable snapshot took approximately:

```text
0.14 ms
```

Qt does not copy all child NumPy arrays while transferring the Python object. Qt delivery is not contributing meaningfully.

## 5. Renderer preparation

Applying the cold 4,453-tile snapshot on the GUI thread took:

| Renderer preparation | Time |
|---|---:|
| Create 4,453 VisPy visual/VBO resources | 3.20 s |
| Insert resources into GPU residency | 1.84 s |
| Remaining key/visibility bookkeeping | approximately 32 ms |
| Total `apply_snapshot()` | **5.29 s** |

The 1.84-second GPU-residency cost is another quadratic bookkeeping issue. Every call to [GPU `retain()`](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/viewer/tiled_points/vispy/residency.py:117) calls a consistency check that rescans all resources retained so far.

Fixing that would reduce cold preparation materially, but the 3.20-second visual-resource creation cost would remain.

## 6. Actual physical rendering

This is the dominant cause of the stuttering.

| Real-canvas operation | Time |
|---|---:|
| First physical draw | **24.03 s** |
| Warm full-view draw, median of seven | **1.087 s per frame** |
| Warm full→quarter visibility activation | 20.0 ms |
| Warm quarter draw, 392 visible tiles | **102 ms per frame** |
| Warm quarter→full activation | 21.6 ms |
| Warm restored full draw | **1.088 s per frame** |

The first draw combines deferred VBO work, shader compilation/linking and 4,453 draw submissions. The benchmark cannot separate those internal OpenGL operations further without lower-level GL tracing.

The warm result is decisive: even after all data, buffers and shaders are resident, drawing 4,453 independent visual nodes still takes approximately one second per frame. A 392-tile viewport takes about 102 ms, or at best approximately 10 frames per second.

`SceneCanvas.render()` includes synchronous framebuffer readback. The empty-canvas baseline was 8.5 ms, so readback contributes a small constant cost but cannot explain the 102–1,087 ms tile-dependent timings.

## 7. Memory amplification in the renderer

The logical AAMP point payload is only 0.69 MiB.

Nevertheless:

| State | Process RSS |
|---|---:|
| Before snapshot renderer resources | 1,006.7 MiB |
| After creating 4,453 resources | 1,241.4 MiB |
| After first physical draw | 1,912.9 MiB |

The renderer added approximately 906 MiB of process RSS. On this Apple system that includes Python, VisPy, driver and unified-memory allocations rather than being a direct VRAM measurement.

The current GPU byte budget counts logical vertex payload bytes. It does not account for thousands of Python objects, shader programs, scene nodes, VBO wrappers and driver resources.

## 8. Warm worker versus warm rendering

Once every AAMP tile was CPU-resident:

| Operation | Time |
|---|---:|
| Warm worker snapshot | 32.5 ms |
| Qt delivery | 0.14 ms |
| Warm renderer activation | 23.3 ms |
| Warm full draw | 1,086.6 ms |

Therefore, the continuing lack of smoothness during zoom is not primarily caused by Zarr after warm-up. It is caused by rendering thousands of separate tile visuals.

## 9. Follow-up investigation: storage-layout alternatives

The current bucket writer appends logical tiles in spatial order and sorts rows inside each tile by `(value_id, point_id)`. The resulting physical order is therefore:

```text
(tile_y, tile_x, value_id, point_id)
```

This is an appropriate primary order for complete-tile and all-value reads. It is a poor primary order for a sparse selected value spanning most of the tissue: one AAMP range occurs in each of 4,453 positive logical tiles, and those small ranges are distributed throughout the bucket point arrays.

### Projected effect of alternative layouts

A read-only scan of the supplied cache's Exact-level range metadata was used to project how many `location` rows AAMP would decode under alternative layouts. These are layout projections, not wall-clock measurements of rebuilt caches.

| AAMP layout | Inner chunks | Decoded `location` rows | Row amplification | `location` shards |
|---|---:|---:|---:|---:|
| Current tile-major layout, 4,096-row chunks | 4,291 | 17,570,347 | 290.4× | 1,067 |
| Current tile-major layout, 256-row chunks | 4,669 | 1,195,145 | 19.8× | 1,067 |
| Current tile-major layout, 64-row chunks | 5,305 | 339,465 | 5.6× | 1,067 |
| Value-major layout inside the existing 69 buckets, 4,096-row chunks | 69 | 282,624 | 4.7× | 69 |
| One value-major array for the complete level, 4,096-row chunks | 16 | 65,536 | 1.08× | 1 |

Smaller chunks greatly reduce decoded content without changing physical row order. They do not reduce the number of sparse selections, and they increase the number of independently handled inner chunks from 4,291 to 5,305 in the 64-row case. They also leave all 1,067 current shards involved because AAMP is distributed throughout the full tissue. The completed smaller-chunk benchmarks did not materially solve the end-to-end latency, so this is no longer a primary implementation direction. Smaller chunks remain useful only as a comparison baseline for a value-major prototype.

### Fewer buckets are not a primary fix for the current layout

Reducing the 69 Exact buckets would reduce store-opening and bucket-dispatch overhead. It would not remove the 4,453 tile/value intervals, materially reduce the fixed-size chunk decode amplification, or change the number of VisPy resources. The measured non-Zarr work for interval resolution, selector construction, grouping, splitting, and ordering was only approximately 80 ms.

A read-only, operating-system-warm concurrency probe also compared the existing sequential bucket loop with a `ThreadPoolExecutor` over independent bucket readers:

| Bucket workers | Observed read times |
|---:|---:|
| 1 | 4.76 s first pass; 2.55–2.63 s repeated |
| 2 | 2.48–2.88 s |
| 4 | 2.57 s |

The first pass is consistent with the application-cold profile, while later passes benefit from warmer operating-system state. Simple 2–4-way cross-bucket threading did not improve the repeated throughput. This is not a universal concurrency qualification, but it shows that dispatching the current sparse decode work concurrently is not the structural solution on this machine.

Fewer buckets become more valuable if rows inside each bucket are value-major: the table shows that a value-major AAMP payload distributed over 69 buckets has a minimum of 69 touched chunks and shards, whereas a single per-level value-major payload needs only 16 chunks for this gene.

### Recommended cache design: dual physical ordering

The strongest cache-side solution is to retain the existing tile-major payload and add a display-only, per-level coordinate payload ordered by:

```text
(value_id, manifest_index, point_id)
```

The two physical representations serve different access patterns:

| Access pattern | Physical payload |
|---|---|
| All values or complete logical tiles | Existing tile-major bucket payload |
| Proper value subset at a level with a sidecar | Per-level value-major coordinate payload |
| Proper value subset at a level without a sidecar | Existing tile-major filtered fallback |

#### What is physically duplicated

This is genuine physical duplication of the coordinate rows. A Zarr array has one physical row order, so the same logical coordinates must be materialized once in tile-major order and once in value-major order. An alternate index into the existing tile-major rows would not solve the decode problem: the index could find AAMP's rows, but those rows would still be scattered across the same tile-major chunks.

It is not necessary to duplicate the complete cache or every per-point field:

| Existing structure | Value-major sidecar | Reason |
|---|---|---|
| `location` | Duplicate and reorder | These are the bytes that must become contiguous for selected-value reads. |
| `value_id` | Omit | The requested value and its catalog interval already identify every returned row. |
| `point_id` | Omit | It can establish deterministic construction order and then be discarded from the display sidecar. |
| tile manifest and transforms | Reuse | `manifest_index` still identifies the tile offset for tile-relative coordinates. |
| value-to-tile catalog | Reuse | The catalog already identifies each value's positive tiles and point counts. |
| bucket sparse-range metadata | Do not duplicate | Retain it initially only for filtered tile-major fallback; the value-major path does not consume it. |

A minimal representation is therefore conceptually:

```text
value_major/
    level_0/
        location              # float32 [N_exact, 2]
        value_point_indptr    # compact start/stop per canonical value
```

Rows in `location` are ordered by `(value_id, manifest_index, point_id)`, but only the coordinates are persisted. `value_point_indptr` gives each canonical value's complete coordinate interval. It is compact because it has one pointer per level/value rather than one pointer per value/tile record.

The cache catalog already persists value-to-tile records in `(level, value_id, manifest_index)` order. The sidecar follows that same record order. The current selected-value index retains the aligned `manifest_index` and `n_points` records only for the active selection. A cumulative sum of those selected counts derives per-record coordinate offsets in memory; a cache-wide persisted or resident `record_point_indptr` is therefore unnecessary. A full-extent AAMP read becomes one contiguous value interval; a rectangular partial viewport becomes a small set of value-major spatial runs rather than one interval in each positive tile. The returned tile-relative coordinates can still be split by catalog record and combined with the existing manifest tile offsets by the snapshot packer.

Both orderings should belong to one atomically published cache generation and share its generation ID, manifest, value vocabulary, and value-to-tile catalog. They are two physical payloads inside one logical cache, not two independently versioned caches that can drift out of sync.

#### Measured storage consequence

The supplied cache currently occupies approximately 1.57 GiB of physical file bytes. Its compressed physical components include:

| Component | Measured physical size |
|---|---:|
| Complete cache | 1.57 GiB |
| `location`, all levels | 1.09 GiB |
| `location`, Exact level only | 0.79 GiB |
| `point_id`, all levels | 300 MiB |
| `value_id`, all levels | 107 MiB |
| ranges plus catalog/metadata | approximately 88 MiB |

If reordered coordinates compress similarly to the current coordinates, the storage projections are:

| Sidecar scope | Estimated added size | Estimated new cache size | Increase |
|---|---:|---:|---:|
| Exact-level `location` only | 0.79 GiB | 2.37 GiB | approximately 50% |
| All-level `location` only | 1.09 GiB | 2.67 GiB | approximately 69% |

These are estimates, not rebuilt-cache measurements. Changing row order can improve or worsen compression, so actual compressed bytes must be recorded by the prototype. The important distinction is that the proposed Exact-only sidecar duplicates approximately 0.79 GiB of coordinates, not the full 1.57 GiB cache and not `point_id` or `value_id`.

#### Persisted versus resident sparse-range policy

The current runtime retains five bucket lookup arrays for every bucket across every level:

```text
tile_offset
ranges/tile_indptr
ranges/value_id
ranges/row_start
ranges/row_count
```

This supplied cache requires 596,026,272 resident NumPy-buffer bytes, approximately 568.4 MiB, for that complete lookup. Exact alone accounts for 295,919,608 bytes, approximately 282.2 MiB. By comparison, the always-resident compact manifest/value-pointer arrays require 821,488 bytes, and the complete AAMP selected-value index across all levels requires 164,964 bytes. The tile and range pointer arrays together account for only 276,112 bytes; the three arrays repeated for every value/tile range account for almost all of the 568.4 MiB.

The initial dual-ordering cache should preserve the existing bucket sparse ranges on disk. They remain necessary because an Exact-only sidecar does not cover a proper-subset request whose point-budget LOD decision selects Bridge or Spatial, and they preserve the existing filtered tile-major fallback and validation contract.

They should no longer be one indivisible, eagerly resident startup index. The runtime policy should be:

| Index data | Residency policy |
|---|---|
| manifest, value pointers, value totals | Always resident |
| active selected value-to-tile records and counts | Resident for the committed selection |
| `value_point_indptr` for sidecar addressing | Always resident; compact |
| tile-major `tile_offset` | Always resident or derived once; compact |
| bucket `ranges/{tile_indptr,value_id,row_start,row_count}` | Load only for selected-value tile-major fallback; retain under a byte-bounded eviction policy |

A value-major request must not load the bucket sparse-range arrays. An all-values tile-major request needs only the complete tile interval and point-level `value_id`; it also does not need the sparse-range arrays. If Bridge or Spatial fallback needs them, load only the chosen level's required buckets—preferably only buckets containing CPU-residency misses—and prevent successive viewports from accumulating every bucket indefinitely.

Keep `ranges/row_start` on disk for the first sidecar slice to avoid combining a cache-format rewrite with the locality experiment. It is a candidate for a later schema simplification because validated ranges partition each tile contiguously: their starts can be reconstructed from `tile_offset` plus a cumulative sum of `range_count`. Removing it requires explicit size, startup, and fallback-read evidence and should be a separate change.

Point-level `bucket/value_id` is distinct from `ranges/value_id`. Keep the point-level array in the tile-major payload initially because all-values rendering needs a colour ID aligned with every coordinate. Proper-subset reads on either physical ordering should construct the output IDs from the known value intervals instead of decoding that point-level array.

#### Explicit physical-payload routing

LOD selection must happen before physical-payload selection. The semantic level still comes from the viewport, selected values, and point budget; the existence of a sidecar must not force Exact when the request requires a coarser level.

After the level is selected, use this deterministic initial routing rule:

| Request after LOD selection | Physical payload | Large bucket sparse-range index |
|---|---|---|
| Over budget | Read neither payload | Not needed |
| All canonical values | Tile-major at the selected level | Not needed |
| Complete-tile or construction access | Tile-major | Not needed for complete row access; publication validation remains separate |
| Proper value subset and sidecar exists for the selected level | Value-major sidecar | Not needed |
| Proper value subset and no sidecar exists for the selected level | Tile-major filtered fallback | Load lazily for required buckets |

Selecting the complete vocabulary is already normalized to the all-values state, so it follows the tile-major branch. For the supplied full-extent, 100,000-point case, AAMP selects Exact with 60,512 points and therefore uses the Exact value-major sidecar; the all-values request selects Spatial level 8 with 100,000 points and therefore uses that level's tile-major payload.

For the initial prototype, any proper subset should use the sidecar when the chosen level has one. This makes routing reproducible and the benchmark interpretable. A later measured cost model may route a dense, near-all-values subset back to tile-major when that is cheaper. Such a model should compare projected touched chunks, physical operations, or selected-row coverage rather than using only the number of selected genes. The backend decision belongs to the generation-bound read plan or cache reader, not the GUI or renderer, and should be made once per snapshot so both paths produce the same logical tile payload and reuse the same CPU-residency contract.

#### Recommended staged prototype

The first cache-side prototype should be deliberately narrow:

1. Build an **Exact-level-only, coordinate-only** value-major sidecar in `(value_id, manifest_index, point_id)` order.
2. Reuse the existing manifest and value-to-tile catalog, persist only compact per-value coordinate pointers, and derive selected per-record offsets from catalog counts.
3. Apply the post-LOD routing table above: proper-subset Exact reads use the sidecar, all-values and complete-tile reads use tile-major, and proper-subset non-Exact reads use the tile-major fallback.
4. Split the current eager bucket lookup policy so value-major and all-values requests do not retain sparse-range arrays. Lazily load and byte-bound only the fallback indexes actually required.
5. Measure construction time, actual compressed size, cold and warm selected-value reads, decoded bytes, physical operations, startup and peak lookup memory, and fallback-index churn for sparse and dense genes at full and partial viewports.
6. Add Bridge or other spatial levels only if runtime evidence shows that selected-value reads at those levels remain an important bottleneck.

Cache construction time is intentionally not an acceptance constraint unless it becomes operationally prohibitive. The main acceptance question is whether the extra approximately 0.79 GiB removes the scattered-decode latency sufficiently to improve interaction after renderer batching.

If that storage increase is unacceptable, lower-storage variants can be evaluated without pretending that an index alone fixes locality:

- build persistent value-major coordinates lazily for selected or frequently used values; AAMP's 60,512 float32 two-dimensional coordinates are only approximately 0.46 MiB raw, excluding metadata;
- store explicitly validated, display-only quantized tile-relative coordinates, for example `uint16`, which halves the raw coordinate width but introduces a precision contract; or
- offer a value-major-only cache profile for workflows that do not require efficient all-value or complete-tile reads, accepting the loss of the current primary access order.

This design deliberately spends cache-construction time and storage to improve runtime behavior. It also creates a path that does not require the complete 568.4 MiB bucket sparse-range lookup to be resident for selected-value reads. The existing small selected-value catalog index can perform discovery, while the value-major payload provides direct coordinate access.

### Complementary cache improvements and comparison baselines

One smaller runtime change remains independently justified:

1. **Do not read point-level `value_id` for proper subsets.**

   `resolve_selected_tile_intervals()` already knows the selected value associated with each range. Reconstructing the aligned IDs from the resolved ranges would remove the measured 1.86-second `value_id` Zarr boundary for AAMP. A one-value renderer could alternatively use a uniform value ID.

The existing smaller-chunk and fewer-bucket benchmarks did not materially solve the end-to-end problem. A 128- or 256-row `location` setting may be retained as a controlled comparison when measuring the value-major prototype, but it is not a recommended implementation slice on its own. Any further comparison must include cache size, inner-chunk index size, physical reads, and wall time; decoded-row reduction alone is not sufficient acceptance evidence.

An uncompressed or memory-mappable display payload is another possible local-cache tradeoff. It would let the operating system service sparse fixed-width rows without codec amplification, at the cost of larger storage and a more local-filesystem-specific backend. The dual value-major Zarr payload should be evaluated first because it fixes locality while retaining the current storage model.

## 10. Follow-up investigation: renderer design

The current renderer creates one `_TiledPointsTileVisualNode`, one visual/program state, and one VBO wrapper per logical tile. In the AAMP profile, the average resource contains only:

```text
60,512 points / 4,453 resources = 13.6 points per resource
```

This is the wrong renderer granularity even though the 512-unit logical tiles remain a useful storage and CPU-residency granularity.

### Recommended renderer: one active snapshot buffer

At the current 100,000-point hard budget, the complete packed vertex payload is at most approximately 1.15 MiB with the existing 12-byte position/value vertex format. Repacking and uploading that bounded payload is preferable to maintaining thousands of tiny GPU resources.

The recommended ownership model is:

```text
logical storage tiles
        -> decoded CPU tile residency
        -> assemble active snapshot vertices
        -> one VisPy visual and one active VBO
```

The initial renderer should use one visual, one shader/program path, and one replaceable VBO. The worker prepares and validates the complete immutable batch before the GUI mutates that VBO. This is the smallest architecture that removes the measured resource and draw-submission bottleneck.

A second fixed ping-pong VBO is an optional hardening or performance follow-up, not an initial requirement. It should be added only if real-canvas measurements show active-buffer update stalls or visible replacement artifacts, or if failure-injection establishes a product requirement to keep the previous GPU payload drawable after `VertexBuffer.set_data()` begins. If needed, both VBOs should belong to the same visual/program; two complete visual nodes are unnecessary.

For the first implementation, tile offsets should be folded into cache-relative assembled positions. This is algebraically identical to the current `a_position + u_tile_offset` shader path and leaves the cache-origin matrix handling unchanged. A snapshot-local origin can be considered later if measurements show a precision need; it should not be mixed into the initial batching change.

The shared palette texture remains applicable. Logical CPU tile residency can also remain unchanged so warm pan and zoom requests reuse decoded data. Per-logical-tile GPU residency is no longer needed; if profiling later justifies GPU-side page reuse, it should use a small fixed pool of larger pages rather than thousands of scene nodes.

This design directly addresses all three renderer symptoms:

- resource creation becomes constant in visual count;
- each accepted snapshot performs one bounded VBO upload, and first draw prepares only one active shader/program path;
- warm frames issue one active point draw instead of thousands of draw submissions.

It also makes the GPU budget representative of the resources it controls. The current logical vertex-byte accounting does not capture per-visual Python, program, scene-node, VBO-wrapper, and driver overhead.

### Concrete refactor in the current code base

The renderer boundary remains the primary place to remove per-tile GPU resources. Decoded CPU tile residency and request scheduling should remain unchanged. The first constant-resource renderer slice can consume `TiledPointsRenderSnapshot.tiles` directly as an instrumented scaffold, but the completed smooth-frame design should extend the snapshot contract with a worker-prepared immutable render batch so NumPy packing does not consume GUI-thread frame time.

The current path is:

```text
cache_session._read_viewport_snapshot()
        -> TiledPointsRenderSnapshot(tiles=...)
        -> composition._on_snapshot_ready()
        -> VispyTiledPointsLayer.apply_snapshot()
        -> one _VispyTileResource / _TiledPointsTileVisual / VBO per tile
```

The first renderer slice may temporarily use this instrumented scaffold:

```text
cache_session._read_viewport_snapshot()
        -> TiledPointsRenderSnapshot(tiles=...)
        -> composition._on_snapshot_ready()
        -> VispyTiledPointsLayer.apply_snapshot()
        -> pack all snapshot tiles into one bounded vertex array on the GUI thread
        -> replace the one snapshot VBO payload
        -> draw the updated snapshot visual
```

The completed target path is:

```text
cache_session._read_viewport_snapshot()
        -> assemble ordered logical tiles on the worker
        -> pack one immutable, bounded render batch on the worker
        -> TiledPointsRenderSnapshot(tiles=..., render_batch=...)
        -> composition._on_snapshot_ready()
        -> VispyTiledPointsLayer.apply_snapshot()
        -> replace the one snapshot VBO payload on the GUI thread
        -> draw the updated snapshot visual
```

The affected responsibilities are:

| File | Current responsibility | Responsibility after the refactor |
|---|---|---|
| `vispy/layer.py` | Creates, retains, hides, and evicts one `_VispyTileResource` per logical tile. | Owns one snapshot visual and one VBO, replaces that VBO from a worker-prepared batch, and commits logical state without a NumPy tile loop. |
| `vispy/visuals.py` | `_TiledPointsTileVisual` packs and uploads one tile and applies `u_tile_offset` in the shader. | A snapshot visual owns one VBO and replaces its complete packed vertex payload; no per-tile visual or tile-offset uniform is needed. |
| `vispy/residency.py` | `_GpuTileResidency` tracks thousands of tile-keyed GPU resources and performs retention/eviction bookkeeping. | Its tile-resource role disappears. Single-buffer byte accounting can be kept directly in the layer or in a small snapshot-buffer helper. |
| `contracts.py` | Carries a tuple of logical render tiles in `TiledPointsRenderSnapshot`. | Defines and validates an immutable, C-contiguous packed render batch and carries it with the generation-bound snapshot. The tile tuple can remain during the migration for CPU-residency identity and diagnostics. |
| `runtime/cache_session.py` | Assembles the ordered logical tile tuple. | Packs the render batch after final ordered tile assembly and before returning an accepted snapshot, with cancellation and generation checks preserved. |
| `runtime/composition.py` | Delivers the snapshot to the layer event. | Forwards the snapshot and packed batch unchanged; it remains unaware of VisPy and VBO ownership. |

#### Snapshot packing

Add one pure NumPy packing helper in a GUI-neutral module such as `viewer/tiled_points/render_batch.py`. It must not live under `viewer/tiled_points/vispy/`: importing that package imports the VisPy layer and its GUI dependencies, which the cache worker should not acquire. The helper should:

1. Allocate one structured array sized exactly to `snapshot.rendered_point_count`.
2. Use the existing vertex fields: `position` as two `float32` values and `value_id` as one `float32`, for 12 bytes per point.
3. Fill consecutive slices in snapshot tile order. For a tile `(tile_x, tile_y)`, calculate cache-relative positions as:

   ```text
   packed.position = tile.location + (tile_x * tile_size, tile_y * tile_size)
   packed.value_id = tile.value_id
   ```

4. Validate that the final write cursor equals the declared rendered point count.
5. Handle an empty snapshot without constructing a dummy per-tile resource.

The simple helper should allocate one primary output and fill slices. Repeated `numpy.concatenate` calls inside the tile loop or constructing one intermediate structured array per tile would reintroduce avoidable allocation and copy overhead. A single whole-snapshot concatenate/repeat strategy is a distinct, measured optimization and remains acceptable if its extra transient memory is justified.

#### Measured packing cost

Packing was benchmarked against the actual full-extent AAMP snapshot already resident in memory:

```text
4,453 tiles
60,512 points
median 5 points per tile
0.693 MiB packed vertex payload
```

| CPU packing operation | Median | p90 |
|---|---:|---:|
| Allocate the structured output only | 0.14 ms | 0.16 ms |
| Iterate tile intervals only | 0.14 ms | 0.15 ms |
| Copy positions and value IDs without offsets | 1.92 ms | 1.99 ms |
| Literal tile loop with coordinate offsets | **7.04 ms** | **7.36 ms** |
| Per-tile `numpy.add(..., out=...)` | 7.24 ms | 7.40 ms |
| One whole-snapshot concatenate/repeat strategy | **2.22 ms** | **2.33 ms** |

The cost is dominated by thousands of NumPy dispatches over very small arrays, not by arithmetic over 60,512 points. Seven milliseconds is negligible relative to the current multi-second renderer preparation and approximately one-second warm draw, so it does not weaken the snapshot-VBO design. It is nevertheless material within a 16.7-millisecond 60 Hz frame budget, especially because VBO upload and drawing still follow it. The completed implementation should therefore keep this work off the GUI thread.

The literal loop remains a reasonable first worker implementation because it is simple, bounded, easy to validate, and permits cancellation checks between groups of tiles. The 2.22-millisecond concatenate/repeat variant uses approximately a few MiB of bounded transient memory at the 100,000-point cap and should remain an optional worker-side optimization if end-to-end worker latency later matters.

#### One-million-point scaling benchmark

A separate in-memory benchmark exercised the same packing functions with synthetic snapshots containing 1,000,000 points. This isolates packing scalability; it does not include Zarr reads, Qt delivery, `VertexBuffer.set_data()`, deferred GPU upload, or physical drawing. The synthetic 60,512-point/4,453-tile baseline measured 6.35 milliseconds for the literal loop, close to the 7.04 milliseconds measured with the actual AAMP snapshot, so it reproduces the relevant packing behavior sufficiently for a scaling experiment.

| Synthetic snapshot shape | Literal tile loop, median | Whole-snapshot concatenate/repeat, median |
|---|---:|---:|
| 1,000,000 points in 1 tile | 5.62 ms | 4.13 ms |
| 1,000,000 points in 4,453 tiles | 11.97 ms | 6.12 ms |
| 1,000,000 points in 7,294 tiles | 15.57 ms | 7.46 ms |
| 1,000,000 points in 100,000 tiles | 145.89 ms | 47.90 ms |

The 7,294-tile case represents the maximum Exact-level tile count in the supplied cache. The 100,000-tile case is deliberately pathological and represents a possible future much larger tissue rather than a layout the current cache can produce.

Packing has two scaling terms:

```text
packing work = per-point memory work + per-tile dispatch work
```

One million points alone does not make packing problematic. With the current cache geometry, the literal worker implementation remains approximately 12–16 milliseconds and the whole-snapshot variant approximately 6–8 milliseconds. Extreme tile fragmentation is the important risk: a point-count budget alone does not fully bound snapshot-preparation work.

The runtime should therefore record both point count and tile count for every packed snapshot. A hard tile-count limit should not be introduced from this synthetic result alone because sparse selected values may legitimately span many tiles. Instrumentation should first establish a point-plus-tile work model that can inform LOD planning or a future preparation-work budget. Packing must remain cancellable so obsolete highly fragmented requests do not occupy the worker unnecessarily.

#### Snapshot visual and VBO ownership

Replace `_TiledPointsTileVisual` with a `_TiledPointsSnapshotVisual`-style implementation that owns:

- one VisPy `VertexBuffer`;
- one shader program and the existing shared palette texture;
- the current uploaded point count;
- a method that replaces the complete vertex payload; and
- deterministic buffer cleanup.

The vertex shader can consume the packed cache-relative position directly and remove `u_tile_offset`. The layer's existing cache-origin transform remains responsible for moving cache-relative coordinates into world space. `_prepare_draw()` should decline drawing when the visual contains zero points.

`VispyTiledPointsLayer` should construct exactly one of these visuals as the child of its existing `Compound` node. That visual owns exactly one VBO in the initial implementation. Each accepted snapshot replaces the complete bounded payload with one `VertexBuffer.set_data()` call and produces one point draw.

Palette changes remain texture updates. Point diameter and opacity update one visual's uniforms and must not cause a vertex re-upload.

#### `apply_snapshot()` transaction

`VispyTiledPointsLayer.apply_snapshot()` should become a bounded single-buffer replacement:

1. Validate the snapshot and reject a point- or byte-budget violation before changing the active visual.
2. Validate that the worker-prepared render batch is immutable, C-contiguous, reconciles with `rendered_point_count`, and has the expected 12-byte vertex format.
3. Preflight that the new payload fits the configured single-buffer byte budget before mutating the VBO.
4. Replace the sole VBO payload with the prepared array using the selected copy-safe lifetime semantics.
5. If `set_data()` raises a Python exception, emit the render error and decline the candidate activation. The previous GPU payload is not guaranteed to remain drawable after mutation has begun.
6. On success, update the point count, set visual drawability according to whether the snapshot is empty, and acknowledge the accepted result.
7. Request one scene update.

Packing failures now occur on the worker before a candidate snapshot is submitted. They should follow the existing worker-error path and leave the active visual untouched. `apply_snapshot()` must not contain the snapshot tile loop in the completed implementation.

This deliberately changes overlap behavior: a changed accepted snapshot performs one bounded upload even when it shares tiles with the preceding snapshot. Warm CPU tile reuse still avoids Zarr reads and decoding. The renderer trades per-tile GPU reuse for a constant scene graph and a single draw submission; at the 100,000-point cap that is the correct trade.

Validation, cancellation, stale-generation rejection, over-budget rejection, and byte-capacity rejection all occur before `set_data()` and therefore continue to preserve the active payload. The initial one-VBO design deliberately does not promise rollback after VBO replacement starts. VisPy defers the actual GPU operation until rendering, so even a ping-pong design would require explicit draw-error handling before it could claim verified GPU-upload rollback.

`_VispyTileResource`, `_GpuTileResidency`, per-tile visibility loops, per-tile LRU retention, per-tile GPU eviction, and renderer-owned active or pending tile-key tuples cease to be part of the normal renderer. The snapshot and runtime already carry generation-bound logical identity; the renderer needs only the active point count and payload metrics. If Slice 12 is later accepted, it introduces a dedicated worker-prepared physical-payload identity rather than reconstructing tile-key identity on the GUI thread. Existing metrics such as resident GPU tile count and GPU eviction count should be replaced with visual count, VBO count, active point count, active bytes, candidate batch bytes, pack time, and upload-staging time. A temporary compatibility alias is acceptable if another internal consumer still reads an old field.

#### Budget implications

With the existing 12-byte vertex format:

```text
100,000 points * 12 bytes = 1,200,000 bytes = approximately 1.15 MiB for the VBO
optional two-VBO ping-pong maximum = approximately 2.29 MiB
1,000,000 points * 12 bytes = 12,000,000 bytes = approximately 11.44 MiB for one VBO or packed batch
optional two-VBO ping-pong maximum at 1,000,000 points = approximately 22.89 MiB
```

Slice 1 may retain the existing `max_gpu_tile_bytes` name only as a temporary implementation scaffold, but its enforcement covers one complete snapshot payload rather than a sum of tile estimates. The bounded worker batch and any temporary CPU copy made by VisPy should be reported separately from logical vertex-payload bytes. At 1,000,000 points, the worker output is approximately 11.44 MiB, the VBO is approximately 11.44 MiB, and `set_data(copy=True)` may temporarily retain another approximately 11.44 MiB CPU staging copy. A whole-snapshot concatenate/repeat implementation also uses temporary coordinate, value-ID, and repeated-origin arrays; its peak must be measured rather than inferred from the final VBO size alone.

Slice 2 must rename this setting to `max_vertex_payload_bytes` across application settings, the layer model, adapter construction, renderer capacity checks, diagnostics, benchmarks, and tests. This is a direct rename, not a compatibility migration: do not retain `max_gpu_tile_bytes` as a deprecated property, constructor keyword, environment/configuration alias, or internal fallback. `max_vertex_payload_bytes` remains an explicit byte bound independent of the hard point-count budget and describes the logical size of one complete packed render batch/VBO payload; it does not claim to bound total GPU, driver, worker, or transient staging memory. Both the 100,000- and 1,000,000-point final VBOs are far below the current 512 MiB default, but an increase to 1,000,000 points still requires deliberate end-to-end memory, upload, and draw evidence rather than only a logical byte calculation.

#### Threading boundary and required implementation slices

VisPy visual and VBO mutation must remain on the GUI thread. Pure NumPy packing should run on the existing cache worker after it has restored the complete tile plan's spatial order and immediately before final snapshot construction. The packed array must own its allocation, be read-only after validation, and cross the Qt object signal by reference. The existing immutable-tile snapshot delivery measured 0.14 milliseconds and did not deep-copy child arrays; delivery must be remeasured with the packed batch rather than merely assumed to remain equivalent.

The work should be separated into two reviewable renderer slices:

1. **Constant GPU-resource topology**

   Replace per-tile visuals and GPU residency with one snapshot visual/program and one replaceable VBO. A temporarily GUI-packed batch is acceptable as an instrumented scaffold in this slice because it isolates resource ownership, shader, coordinate, colour, and real-canvas behavior. Acceptance requires exactly one visual, one VBO, one payload replacement per accepted snapshot, and one point draw submission. Capacity and validation failures must still preserve the active payload because they are rejected before VBO mutation. This slice alone is not the completed smooth-frame implementation.

2. **Worker-prepared render batch**

   Move the pure NumPy helper to the worker path, extend the generation-bound render contract, preserve cancellation before and during packing, and make the GUI consume only the prepared batch. Packing failures and obsolete requests must not reach VBO upload. This is a required smooth-frame completion slice, not an optional optimization.

The slices may be implemented together if that proves simpler, but the roadmap should preserve their separate acceptance evidence. The GUI-side scaffold must not become the final architecture merely because its cost is much smaller than the renderer it replaces.

An optional third slice may introduce a second fixed VBO owned by the same snapshot visual/program. It is justified only by measured active-buffer synchronization stalls, visible replacement artifacts, or a required and tested recovery policy for staging/draw failures. The acceptance evidence must show that the second VBO solves the observed problem. It must not introduce a second visual or a second draw submission merely to obtain ping-pong storage.

#### Required verification changes

The packing, delivery, and renderer tests should be rewritten around snapshot-batch semantics:

- worker-packed positions, snapshot order, and value IDs are correct across multiple tiles;
- the packed batch owns a C-contiguous allocation, is read-only after construction, and reconciles with the snapshot count and byte budget;
- packing failures, cancellations, and obsolete request generations do not submit a renderer candidate;
- queued Qt delivery of a maximum-budget packed batch does not deep-copy its vertex storage;
- completed GUI activation performs no NumPy tile-packing loop;
- one accepted non-empty snapshot causes one VBO payload upload;
- palette, opacity, and point-diameter updates do not upload vertex data;
- validation and capacity failures preserve the previous active payload because they occur before VBO mutation;
- an injected synchronous `set_data()` failure reports an error and does not commit candidate logical state, without asserting that the previous GPU payload remains drawable;
- an empty accepted snapshot hides the active draw without creating resources;
- closing the layer releases the one visual and VBO exactly once;
- a real-canvas test verifies position and colour, including a large cache origin; and
- the former overlapping-tile GPU-reuse test is replaced by an assertion that visual count and VBO identity remain constant across changing snapshots.

Before increasing the current 100,000-point budget, add a 1,000,000-point scalability case that measures:

- worker packing across representative dense, current-maximum-tile, and highly fragmented shapes;
- cancellation latency for an obsolete in-progress pack;
- queued Qt delivery and whether the packed allocation is copied;
- single-VBO replacement staging and deferred upload;
- repeated replacements with changing payload sizes, including driver stalls or blank/corrupted frames;
- first and warm physical draws; and
- representative point diameters and overlap, because fragment overdraw may dominate once draw-submission overhead is removed.

The cache-to-canvas benchmark should stop reporting per-tile renderer construction as the primary renderer metric. It should separately measure worker snapshot assembly, worker packing, queued packed-batch delivery, single-VBO replacement staging, activation, first draw, and warm draw, while confirming that exactly one point visual is submitted. It should also look for frame stalls or blank/corrupted frames across repeated payload-size changes; those measurements determine whether optional ping-pong storage is justified. The acceptance target is not merely a lower construction time: first draw and subsequent camera-interaction frames must lose their dependence on the number of logical tiles in the snapshot, and GUI-thread activation must contain no tile-proportional CPU packing.

The synthetic scaling benchmark proves only that packing 1,000,000 points is reasonable for the current tile geometry when performed on the worker. It does not establish that uploading or physically drawing 1,000,000 points is smooth.

## 11. Comprehensive implementation plan from the current code base

This section translates the preceding findings into ordered, reviewable implementation slices. Each slice has one primary responsibility, keeps the experimental renderer opt-in, and must leave the repository in a working state with focused tests and benchmark evidence. Cache-format work is deliberately separated from renderer work so performance changes can be attributed to one boundary at a time.

### Starting point and non-negotiable constraints

The current working tree has the following starting architecture:

- `ViewerWidget` uses the original in-memory points backend by default. The tiled-cache backend is selected for the lifetime of a new widget only when `experimental_tiled_points=True` is passed directly or `NAPARI_HARPY_EXPERIMENTAL_TILED_POINTS=1` is set before the Viewer widget is constructed.
- The cache-backed path is wired end to end through `TiledPointsController`, the adapter, the napari layer, the viewport coordinator, the worker-owned cache session, and the VisPy renderer.
- Logical storage tiles and decoded CPU tile residency are useful and remain part of the design.
- The renderer still owns one VisPy visual and VBO per logical tile.
- Cache startup still loads every bucket sparse-range lookup index across every level.
- Proper-subset tile-major reads still decode both `location` and point-level `value_id`.
- The cache contains only the existing tile-major physical payload; no value-major sidecar exists yet.

The following constraints apply to every slice:

1. The in-memory backend remains the default and must not construct or bind a cache controller, open cache metadata, or change behavior because a tiled slice landed.
2. Cache failures in opt-in mode remain visible. Do not silently switch an active tiled widget to the in-memory backend.
3. The backend remains fixed for a widget lifetime. Live backend switching is outside this plan.
4. Keep the current 100,000-point hard render budget throughout these slices. A larger budget has a separate evidence gate.
5. Preserve generation checks, latest-only activation, cancellation, exact point-budget enforcement, palette semantics, transforms, and deterministic cleanup.
6. Keep VisPy and VBO mutation on the GUI thread. Zarr access, logical tile assembly, and final NumPy snapshot packing belong to the worker.
7. Do not combine a cache-schema change with a renderer-ownership change in one review slice.
8. Run focused unit tests for the changed boundary and the cache-to-canvas benchmark for performance claims. Real-canvas tests remain explicitly gated where required by the existing test infrastructure.

### Delivery sequence

| Slice | Primary result | Depends on | Status after merge |
|---|---|---|---|
| 0 | Preserve the opt-in boundary and freeze the baseline | Current code | Completed starting point |
| 1 | Replace per-tile GPU resources with one visual and one VBO | Slice 0 | Dominant draw-submission problem removed; temporary GUI packing allowed |
| 2 | Move complete render-batch packing to the worker | Slice 1 | Completed smooth-frame renderer architecture |
| 3 | Make CPU tile retention linear for the no-eviction case | Slice 0 | Cold CPU assembly defect removed |
| 4 | Stop decoding point-level `value_id` for proper subsets | Slice 0 | One of two selected-value Zarr reads removed |
| 5 | Add an Exact-only coordinate value-major sidecar to the cache format and writer | Slice 0 | New payload is constructible and validated but not yet used by the viewer |
| 6 | Route proper-subset Exact reads through the sidecar | Slices 4 and 5 | Sparse selected values gain contiguous coordinate reads |
| 7 | Replace eager sparse-range residency with lazy byte-bounded fallback indexes | Slice 6 | Startup time and lookup RSS are reduced |
| 8 | Run the integrated acceptance matrix and decide sidecar expansion | Slices 1–7 | Evidence-backed decision on Bridge/spatial sidecars |
| 9 | Add viewport debounce only if dispatch churn remains material | Slice 8 | Conditional reduction of obsolete cold reads |
| 10 | Evaluate optional ping-pong storage and a larger point budget | Slice 8 | Conditional hardening/scaling work, not part of the initial solution |
| 11 | Replace implicit initial selection with explicit coordinator arming | Slice 0 | No unconfigured or accidental all-values first viewport |

Slices 1 and 2 form one renderer milestone. Slice 1 may be reviewed and measured independently, but Slice 2 is required before the renderer work is considered complete. Slices 5 and 6 form one cache-locality milestone: publishing a sidecar that no read path consumes is useful only as a short-lived, testable construction boundary.

### Slice 0 — Preserve the opt-in boundary and freeze the baseline

This is the completed starting checkpoint, not a new performance implementation.

**Scope**

- Keep ordinary `ViewerWidget(...)` and `Interactive(sdata)` construction on `PointsController` unless the environment opt-in was set before Viewer construction.
- Keep explicit `ViewerWidget(..., experimental_tiled_points=True)` available for tests and direct programmatic use.
- Record the selected backend through `points_visualization_backend` for diagnostics.
- Preserve the current AAMP baseline report and benchmark command parameters.

**Required regression coverage**

- Default construction selects `PointsController` and executes the existing `load_value_source()` and `load_selection()` callbacks.
- Environment or constructor opt-in selects `TiledPointsController` and executes descriptor loading and `apply_selection()`.
- An explicit constructor value overrides the environment.

There is intentionally no dedicated regression test for changing the environment after a Viewer widget already exists. The environment opt-in is temporary, is sampled only during widget construction, and will be removed in a future version. Backend lifetime is therefore kept as an implementation contract, while the required coverage remains focused on construction-time selection and the two real callback paths.

**Exit condition**

The existing `tests/test_viewer_widget.py` backend tests remain green through all subsequent slices. Every performance benchmark states explicitly that the tiled backend is enabled.

### Slice 1 — Constant GPU-resource topology

**Status: Implemented**

This slice addresses the dominant measured problem: 4,453 visuals, VBO wrappers, shader/program paths, and draw submissions for 60,512 points.

**Current-to-target interpretation**

This is a renderer-ownership change, not a cache-layout or logical-tile change. The worker continues to return a `TiledPointsRenderSnapshot` containing the complete ordered tuple of logical `TiledPointsRenderTile` objects for the accepted viewport. CPU tile residency also remains unchanged.

Today, `VispyTiledPointsLayer.apply_snapshot()` looks up every logical tile in `_GpuTileResidency`, creates a `_VispyTileResource` for every residency miss, inserts that resource's visual and VBO under the `Compound` root, and then loops over old and new tile keys to change visibility. Consequently, thousands of logical storage tiles become thousands of independently traversed VisPy resources.

After this slice, the ownership is fixed at renderer construction:

```text
VispyTiledPointsLayer
└── Compound root
    └── snapshot visual
        ├── one shader/program path
        ├── one VertexBuffer
        └── one shared palette-texture binding
```

The snapshot may still contain 4,453 logical tiles, but those tiles are no longer GPU ownership units. Every accepted nonempty snapshot is packed into one complete vertex payload and replaces the contents of the same stable VBO. Overlapping logical tiles between successive viewports are therefore not reused as independent GPU buffers; full-payload replacement is deliberate because it gives constant visual, VBO, and draw-submission counts. The renderer does not retain or reconstruct active or pending tile-key tuples. A future exact-reuse implementation must consume the dedicated physical-payload identity specified by Slice 12.

The packed `a_position` values are relative to the shared cache origin. Packing adds each tile's `(tile_x * tile_size, tile_y * tile_size)` offset to its tile-local coordinates, which allows the per-visual `u_tile_offset` uniform to disappear. These are not large absolute world coordinates: the existing float64 root transform continues to add the shared cache origin and apply the napari layer transform.

For this slice only, the one-array tile loop runs synchronously inside GUI-thread snapshot activation. This temporary scaffold isolates and validates the renderer topology change. Slice 2 moves the same packing helper to the worker and transports an immutable render batch; it does not introduce a second packing implementation.

The single-VBO failure boundary must also be explicit. Validation, byte-capacity checking, and packing happen before `VertexBuffer.set_data()`, so failures in those phases leave the preceding payload untouched. A candidate is acknowledged only after `set_data()` succeeds. If `set_data()` itself raises after mutation has begun, report the render failure and decline the candidate, but do not claim that the preceding GPU contents remain drawable. A second or ping-pong VBO remains an evidence-gated hardening option rather than part of this slice.

**Production changes**

1. In `viewer/tiled_points/vispy/visuals.py`, replace `_TiledPointsTileVisual` with a snapshot visual that owns:
   - one shader/program path;
   - one shared palette texture binding;
   - one `VertexBuffer`;
   - one current point count; and
   - deterministic `replace_vertices()`, empty-state, uniform-update, and `close()` behavior.
2. Remove `u_tile_offset` from the vertex shader. Pack cache-relative positions before upload and keep the existing cache-origin transform unchanged.
3. In `viewer/tiled_points/vispy/layer.py`, construct exactly one snapshot visual under the existing `Compound` root during renderer initialization.
4. Change `apply_snapshot()` to:
   - validate generation, point count, vertex format, and byte capacity before VBO mutation;
   - pack the accepted snapshot into one bounded vertex array as a temporary scaffold;
   - call `VertexBuffer.set_data()` once;
   - acknowledge the candidate only after synchronous staging succeeds; and
   - request one scene update.
5. Remove `_GpuTileResidency`, `_VispyTileResource`, per-tile visibility changes, and per-tile GPU LRU behavior from the normal renderer path. Do not spend a separate slice optimizing the quadratic GPU consistency scan because this slice removes its ownership model.
6. Do not retain or reconstruct active or pending tile-key tuples in the renderer. Slice 12 must add its dedicated physical-payload identity only if its evidence gate is met.
7. Retain the current `max_gpu_tile_bytes` name only for the Slice 1 scaffold and enforce it against the single candidate vertex payload rather than a sum of tile resources. This temporary implementation state is not a compatibility promise; Slice 2 removes the old name completely.
8. Replace GPU tile metrics with visual count, VBO count, active point count, active vertex bytes, payload-replacement count, and synchronous staging time. Compatibility aliases may exist for one transition only if a current internal consumer needs them.
9. Preserve palette, opacity, point-diameter, blending, large-origin transforms, empty snapshots, close behavior, and render-error signaling.

Introduce the canonical vertex dtype and one pure helper in a new GUI-neutral `viewer/tiled_points/render_batch.py` for this scaffold. The helper runs on the GUI thread only in this slice and is moved, not duplicated, in Slice 2.

**Focused tests**

- Rewrite `tests/viewer/tiled_points/vispy/test_layer.py` around one stable visual and VBO identity across changing snapshots.
- Assert one payload replacement for a nonempty accepted snapshot and no replacement for over-budget, invalid, or capacity-rejected snapshots.
- Preserve stale request/selection rejection at the coordinator/layer integration boundary and prove that a stale result never calls VisPy `apply_snapshot()`.
- Assert that palette, opacity, and point-diameter changes do not replace vertex data.
- Assert that an empty accepted snapshot suppresses drawing without allocating another resource.
- Inject a synchronous `set_data()` exception and assert render-error emission and no logical candidate commit. Do not claim that the previous GPU payload remains drawable after mutation starts.
- Replace the overlapping-tile GPU-reuse test with constant-resource assertions.
- Update `tests/viewer/tiled_points/vispy/test_real_canvas.py` to verify position and palette-indexed colour with one visual, including a large cache origin.
- Delete or quarantine `vispy/test_residency.py` only when no production consumer remains.

**Benchmark evidence**

Update `scripts/benchmark_tiled_points_cache_to_canvas.py` to report visual count, VBO count, packed bytes, payload staging, first draw, and warm draw. Retain the old baseline field names only long enough to compare reports.

On the supplied full-extent AAMP case:

- visual count is exactly one;
- VBO count is exactly one;
- one accepted snapshot causes one point draw submission;
- warm draw time no longer scales with the 4,453 logical tile count;
- warm full-view draw improves by at least an order of magnitude from the 1.087-second baseline; and
- process RSS no longer grows by hundreds of MiB merely because thousands of logical tiles are present.

If one visual still misses an interactive frame target, profile point/fragment cost and VisPy staging before changing the architecture. Do not reintroduce per-tile visuals.

**Exit condition**

The renderer has constant GPU-resource topology and correct rendering. The remaining known defect is that tile-proportional packing still occurs on the GUI thread; this is explicitly resolved by Slice 2.

### Slice 2 — Worker-prepared immutable render batch

This slice completes the renderer milestone by removing the tile loop from GUI activation.

**Production changes**

1. Extend and reuse the Slice 1 helper in `viewer/tiled_points/render_batch.py` so it:
   - allocates exactly one primary structured array;
   - fills consecutive slices in snapshot tile order;
   - folds `(tile_x * tile_size, tile_y * tile_size)` into `float32` positions;
   - writes canonical value IDs as exactly representable `float32` values;
   - validates the final cursor and maximum palette ID;
   - accepts an optional cancellation callback and checks it periodically between tile groups; and
   - returns a C-contiguous, owning, read-only allocation, including a valid empty batch.
2. In `viewer/tiled_points/contracts.py`, add an immutable render-batch contract and carry it on `TiledPointsRenderSnapshot`. Validate dtype, shape, ownership, contiguity, immutability, and byte count. For a within-budget snapshot, reconcile the batch count with `estimated_point_count` and the logical tiles; an over-budget metadata-only snapshot carries an empty batch even when its estimate is nonzero.
3. In `runtime/cache_session.py`, pack after the complete logical tile tuple has been restored to plan order and before constructing the final accepted snapshot. Check cancellation before packing, during fragmented packing, and after packing.
4. Keep the logical tile tuple during this migration because it still supplies CPU-residency identity and diagnostics. The renderer must consume only `render_batch`.
5. Keep `runtime/composition.py` transport-only: it forwards the generation-bound snapshot and batch without knowing the vertex format or VisPy ownership.
6. In `vispy/layer.py`, remove the scaffold packing call. GUI activation validates the already prepared batch, preflights capacity, stages one VBO payload, commits logical state, and updates the scene.
7. Initially use the copy-safe `VertexBuffer.set_data()` lifetime behavior already relied on by the renderer and report any CPU staging copy separately. A later zero-copy change requires explicit lifetime and deferred-upload evidence.
8. Rename `max_gpu_tile_bytes` to `max_vertex_payload_bytes` throughout `TiledPointsApplicationSettings`, `TiledPointsLayerModel`, application-adapter construction, renderer capacity validation, diagnostics, benchmarks, and tests. Remove the old name outright: do not add a deprecated constructor keyword, property, configuration alias, or fallback. Continue to enforce the renamed setting against the logical byte size of one complete packed vertex payload, separately from the hard point-count budget and from measured transient memory.

**Focused tests**

- Add direct packing tests for multiple tiles, sparse tiles, ordering, offsets, values, empty batches, large cache-relative positions, incorrect declared counts, and palette overflow.
- Assert the packed batch owns one read-only C-contiguous allocation and uses the expected 12 bytes per point.
- Add cache-session tests proving that resident and newly read tiles are packed in final plan order.
- Add cancellation tests before and during a highly fragmented pack.
- Assert that obsolete request generations never submit their completed batch to the renderer.
- Add a queued Qt-delivery test for a maximum-budget batch and verify that the vertex allocation identity is preserved across signal delivery.
- Instrument `apply_snapshot()` in tests and assert that it performs no tile iteration or NumPy coordinate packing.
- Update configuration, model, adapter, renderer, error-message, and benchmark tests to use only `max_vertex_payload_bytes`; no compatibility test for the removed `max_gpu_tile_bytes` input is required.

**Benchmark evidence**

The cache-to-canvas report must separate:

```text
logical snapshot assembly
worker render-batch packing
queued Qt delivery
GUI VBO staging
first physical draw
warm physical draw
```

Record both point count and logical tile count for every packed batch. Re-run the actual AAMP case and the synthetic 1,000,000-point cases. At the current 100,000-point cap, GUI activation must contain no tile-proportional work. Qt delivery must remain effectively reference transfer rather than a deep copy.

**Exit condition**

The completed path is worker tiles → immutable packed batch → queued delivery → one GUI-thread VBO replacement → one draw. No normal renderer code creates a resource per logical tile. `max_vertex_payload_bytes` is the sole production setting name for the complete packed-payload byte bound, and no `max_gpu_tile_bytes` input alias or property remains.

### Slice 3 — Linear CPU tile retention

This slice removes the measured approximately 852-millisecond CPU-residency defect without changing residency semantics.

**Production changes**

1. In `runtime/residency.py`, make `_evict_until_fits()` return before materializing the key collection when the requested payload already fits.
2. When eviction is required, traverse the LRU once, skip protected keys, and stop immediately after enough bytes are available.
3. Keep byte reconciliation outside the per-tile insertion loop. Debug-only deep consistency checks must not rescan the complete residency after every inserted tile in production.
4. Preserve oversized-transient behavior, protected-active entries, deterministic LRU order, duplicate-key replacement, and exact byte accounting.

**Focused tests**

- Extend `runtime/test_residency.py` with no-eviction bulk retention, protected eviction, insufficient-capacity, replacement, and oversized-transient cases.
- Add an instrumentation-based regression test showing that a fitting insertion does not enumerate existing keys.
- Retain exact accounting assertions after complete operations.

**Benchmark evidence**

Re-run retention with 4,453 AAMP-shaped tiles and a budget large enough to avoid eviction. The benchmark-machine target is to reduce this phase from approximately 852 milliseconds to tens of milliseconds, with near-linear scaling when the tile count doubles.

**Exit condition**

CPU retention is no longer visible as a major cold-snapshot phase, and warm CPU tile reuse remains unchanged.

### Slice 4 — Eliminate point-level `value_id` reads for proper subsets

This slice reduces selected-value tile-major IO before introducing a new physical layout.

**Production changes**

1. Extend the internal selected-range result in `storage/bucket_reader.py` so every resolved interval retains the canonical value ID that produced it.
2. For proper-subset requests, read only `location` from Zarr and construct the aligned output IDs in memory from interval value IDs and row counts.
3. Preserve deterministic selected-value and tile order across multi-value requests and bucket batching.
4. Continue reading point-level `value_id` for all-values or complete-tile display requests; those rows contain multiple values and the tile-major payload remains their canonical source.
5. Keep the external `_PointDisplayPayload`, `_TileReadResult`, CPU residency, snapshot, and renderer contracts unchanged.

**Focused tests**

- Update `test_bucket_reader.py` to patch the point-level `value_id` array so any access fails during proper-subset reads.
- Cover one value, several nonadjacent values, adjacent ranges, missing values, multiple tiles in one bucket, and restoration of request order.
- Compare reconstructed IDs and coordinates byte-for-byte with the existing canonical result on fixtures.
- Assert that all-values reads still access and return point-level IDs.

**Benchmark evidence**

For full-extent AAMP, point-level `value_id` Zarr calls must fall from 69 to zero while the returned 60,512 IDs remain correct. Report the remaining `location` time independently; this slice is expected to remove the measured approximately 1.86-second value-ID boundary but does not fix scattered coordinate decoding.

**Exit condition**

Proper-subset tile-major fallback performs coordinate-only physical reads and synthesizes IDs from already validated metadata.

### Slice 5 — Exact-level value-major sidecar schema and writer

This slice makes the new physical ordering constructible, atomically published, and independently validated. It does not route viewer reads to it yet.

**Schema and compatibility decisions**

1. Introduce an explicit sidecar descriptor in root cache metadata rather than inferring capability from directory presence. It records covered levels, row ordering, coordinate dtype, dimensionality, chunk/shard settings, and sidecar schema version.
2. Bump the cache schema for sidecar-aware generations and keep a compatibility reader for the current tile-major-only generation. An older cache is interpreted as having no sidecar and continues to use fallback routing.
3. Store the initial sidecar under one unambiguous generation-owned path such as:

   ```text
   value_major/
       level_0/
           location
           value_point_indptr
   ```

4. Persist only Exact `location` and compact `value_point_indptr`. Do not duplicate point-level `value_id`, `point_id`, the manifest, or the value-to-tile catalog.
5. Make Exact sidecar construction an explicit builder option during the prototype. Do not silently add approximately 0.79 GiB to every cache until Slice 8 accepts the tradeoff.

**Writer changes**

1. Add a dedicated storage writer rather than extending `_BucketWriter` with a second unrelated row order.
2. Build inside the same unique staging generation as the tile-major payload and before `_validate_staged_cache()` and publication.
3. Write rows in `(value_id, manifest_index, point_id)` order. Reuse existing per-value catalog records and their counts to determine output intervals. `point_id` establishes deterministic order during construction but is not persisted in the sidecar.
4. Construct the sidecar out of core. Bound temporary memory by a configured construction batch; cache-construction time is secondary to runtime locality but unbounded RAM is not acceptable.
5. Reconcile every value pointer interval with the Exact catalog count and reconcile the final pointer with the Exact manifest total.
6. Include sidecar files in staging validation and atomic publication. Any sidecar write or validation failure must leave the preceding completed generation recoverable.
7. Thread the explicit Exact-sidecar option through the public builder configuration and `scripts/build_tiled_points_cache_variant.py` so the supplied-cache prototype is reproducible from one recorded command.

**Focused tests**

- Add small-fixture tests for ordering, value pointers, empty values, several tiles per value, deterministic point order, chunk boundaries, and final row-count reconciliation.
- Add corruption tests for metadata, dtype, shape, pointer monotonicity, pointer terminal value, and coordinate count.
- Extend builder and staging-validation tests to cover successful publication, rollback on sidecar failure, and tile-major-only compatibility.
- Verify that an unrecognized sidecar schema is rejected rather than ignored.

**Construction evidence**

Build the supplied cache with an Exact coordinate sidecar and record construction duration, peak RSS, sidecar logical bytes, actual compressed physical bytes, total cache size, and compression ratio. Construction duration is reported but is not a rejection criterion unless operationally prohibitive.

**Exit condition**

A completed cache generation can truthfully advertise and validate an Exact coordinate value-major sidecar, while old tile-major-only caches remain readable.

### Slice 6 — Post-LOD physical-payload routing and sidecar reads

This slice realizes the cold-read improvement while preserving one logical tile/snapshot contract above the reader.

**Production changes**

1. Keep `select_level()` unchanged and execute it before physical-payload selection.
2. Extend the generation-bound viewport plan with an explicit physical route:

   ```text
   over budget                         -> no payload
   all values                          -> tile-major complete-tile reads
   proper subset + sidecar at level    -> value-major coordinate reads
   proper subset + no sidecar at level -> tile-major filtered fallback
   ```

3. Make the route decision once per plan in `_PointsCacheReader`; do not decide independently in the GUI, cache session, or per bucket.
4. Add a dedicated sidecar reader that opens the compact per-value pointers and `location` array. Full-extent one-value reads become one value interval. Partial viewports derive only the selected value/manifest-record runs needed for CPU-residency misses.
5. Use the existing selected-value index's aligned `manifest_index` and `n_points` records. Derive per-record sidecar offsets with cumulative counts; do not introduce a cache-wide resident `record_point_indptr`.
6. Split returned coordinate runs back into the same ordered logical `_TileReadResult` values used by tile-major reads. Construct `value_id` arrays from the known value intervals.
7. Keep `_read_viewport_snapshot()`, CPU tile residency, render-batch packing, composition, and VisPy unaware of which physical payload supplied a tile.
8. Expose route, sidecar selection count, touched chunks/shards, selected rows, decoded rows, and physical bytes in benchmark diagnostics.

**Focused tests**

- Prove that LOD is identical with and without a sidecar.
- Cover proper-subset Exact sidecar routing, all-values Exact tile-major routing, proper-subset Bridge/spatial fallback, and old-cache fallback.
- Compare sidecar and tile-major results for one value, several values, full extent, partial viewport, CPU-residency misses, and empty intersections.
- Assert identical tile keys, tile order, coordinates, value IDs, estimated counts, omitted values, and cache-origin behavior.
- Patch sparse bucket-range loading to fail and prove it is not touched by a sidecar request.
- Exercise cancellation and stale generations during a multi-run sidecar read.

**Benchmark evidence**

For full-extent AAMP at Exact:

- the route is `value_major_subset`;
- selected coordinate rows remain 60,512;
- touched `location` chunks fall from 4,291 toward the projected 16 rather than remaining proportional to positive tiles;
- no point-level `value_id` array is decoded;
- no bucket sparse-range array is needed for the request; and
- cold selected-value payload time improves materially from the 4.14-second aligned-array baseline. A practical prototype target is below one second on the same benchmark machine and filesystem state, but the report must retain raw chunk, byte, and call evidence rather than accepting wall time alone.

Also benchmark a dense gene, multiple genes, a partial viewport, and all values to ensure the new route does not regress the tile-major use cases.

**Exit condition**

The reader chooses physical locality after semantic LOD selection and returns the existing logical tile contract. Exact sparse-value reads use the sidecar; uncovered levels fall back correctly.

### Slice 7 — Lazy, byte-bounded sparse-range fallback indexes

This slice removes the approximately 8.1-second startup and 568.4-MiB eager lookup policy.

**Production changes**

1. Split bucket addressing into:
   - compact complete-tile offsets; and
   - large sparse selected-value ranges.
2. Keep compact manifest, value pointers, totals, sidecar pointers, and complete-tile addressing resident. Prefer deriving bucket-local complete-tile offsets once from manifest order and `n_points` so startup does not have to open every bucket merely to read `tile_offset`; validate them against a bucket when that bucket is opened. Do not treat all five current bucket lookup arrays as one indivisible load unit.
3. Remove the unconditional all-level `project_bucket_lookup_index_bytes()` and `load_bucket_lookup_indexes()` sequence from `_TiledPointsCacheWorker.start()`.
4. Sidecar requests load no bucket sparse ranges. All-values complete-tile requests also load no sparse ranges.
5. Before a selected-value tile-major fallback read, identify only the required bucket keys and load their sparse ranges on the worker thread.
6. Add a byte-bounded LRU for sparse-range indexes. Account exact NumPy bytes, evict only inactive indexes, and prevent successive viewports and levels from accumulating every bucket indefinitely.
7. Retain open-reader and sparse-index residency as separate policies. Opening lightweight Zarr metadata must not imply retaining its large lookup arrays.
8. Rename the runtime setting to describe a sparse-range-index byte cap, or provide one compatibility alias while migrating `TiledPointsApplicationSettings` and `_CacheSessionSettings`. It must no longer mean that every bucket index must fit simultaneously.
9. Preserve persisted `ranges/row_start` in this slice. Any schema simplification is later, separate work.
10. Replace startup progress/status that assumes a complete index load with ready-state and on-demand fallback-index diagnostics.

**Focused tests**

- Session startup reaches ready without loading sparse ranges.
- Sidecar and all-values requests leave sparse resident bytes at zero.
- Proper-subset fallback loads only required buckets and reuses them on a warm request.
- The LRU evicts deterministically under its byte cap and never evicts an index in active use.
- Load failure leaves preceding resident indexes valid and reports the viewport failure without corrupting session state.
- Repeated level/view changes remain within the configured cap.
- Old tile-major-only caches still function through lazy fallback.

**Benchmark evidence**

Report startup metadata time, time to ready, compact resident bytes, open bucket readers, sparse resident bytes, on-demand load time, eviction count, and peak RSS. For an Exact-sidecar AAMP startup, sparse resident bytes must remain zero and the previous 568.4-MiB eager allocation must disappear. Fallback benchmarks must demonstrate a stable memory ceiling across repeated viewports.

**Exit condition**

Large sparse ranges are a bounded fallback resource rather than a mandatory session-wide startup index.

### Slice 8 — Integrated acceptance matrix and sidecar expansion decision

This slice consolidates evidence; it is not permission to broaden the format automatically.

**Benchmark matrix**

Run the same cache generation and renderer across:

- sparse one-value AAMP, full extent and partial viewport;
- at least one dense value;
- several selected values;
- all values;
- Exact, Bridge, and representative spatial LOD decisions;
- cold application caches and repeated warm CPU-resident requests;
- full → partial → full camera transitions;
- 100,000-point real-canvas rendering; and
- synthetic 1,000,000-point packing only, without raising the product budget.

For every case report:

```text
LOD and physical route
planned/returned tile and point counts
physical calls, chunks, shards, decoded rows and bytes
fallback-index load/resident/eviction metrics
CPU residency lookup/read/retain time
viewport events, dispatched requests and accepted snapshots
accepted snapshots whose physical render-payload identity matches the active payload
worker pack time and peak transient bytes
Qt delivery time and allocation identity
VBO staging time and active bytes
visual/VBO/draw count
first and warm physical draw
process RSS at startup, snapshot, staging and first draw
```

**Decision rules**

- Keep the Exact sidecar only if selected-value interaction improves enough to justify its measured compressed size.
- Add a Bridge or spatial sidecar only when benchmark traces show that proper-subset reads at that level remain a material interaction bottleneck and projected storage is acceptable.
- Do not add all-level sidecars merely because construction is available.
- Do not substitute smaller chunks, fewer buckets, or cross-bucket threading for the sidecar unless new end-to-end evidence contradicts the existing results.
- Do not increase the point budget based on packing time alone.

**Exit condition**

Publish one comparison report containing the pre-change baseline and each accepted slice. Record explicit keep, revise, or reject decisions for Exact sidecar defaulting and any further levels.

### Slice 9 — Conditional viewport debounce

Debounce is deliberately last because it avoids work but does not make an accepted request cheaper.

**Entry condition**

Proceed only if Slice 8 instrumentation shows that rapid camera gestures still dispatch multiple physical reads that become obsolete despite the existing one-active/one-latest-pending mailbox.

**Production changes if justified**

1. Add a short configurable GUI-thread single-shot timer at the coordinator submission boundary. Do not place timers or sleeps on the cache worker.
2. Advance request generation immediately when a viewport event arrives so older results become stale immediately, but delay physical dispatch until the debounce settles.
3. Do not debounce value-selection changes, startup readiness, explicit refresh, or an already completed isolated request unless measurements justify that latency.
4. Preserve one-active/one-latest-pending behavior, selection-generation rules, close behavior, and failure recovery.
5. Report events received, requests dispatched, requests superseded before dispatch, active reads completed stale, and isolated-request added latency.

**Focused tests**

- Deterministic fake-clock tests for one isolated event, a rapid burst, events arriving during an active read, selection changes during the timer, failure, and close.
- Assert that the final viewport of a burst is dispatched exactly once and older generations cannot activate.

**Acceptance evidence**

Use recorded camera traces rather than synthetic event counts alone. The debounce must materially reduce obsolete cold reads without making an isolated pan or zoom feel delayed. If it does not, retain the current mailbox policy and reject this slice.

### Slice 10 — Optional hardening and scaling gates

These are explicit decision gates, not assumed follow-up work.

#### Optional second VBO

Add a second VBO to the existing single snapshot visual/program only if repeated real-canvas replacements show active-buffer synchronization stalls, blank/corrupted frames, or a product requirement for stronger post-mutation recovery. Evidence must show that ping-pong storage fixes the observed issue. Keep one visual and one draw submission.

#### Optional 1,000,000-point product budget

Do not raise the budget until end-to-end tests cover worker packing, Qt delivery, VBO staging/deferred upload, repeated changing payload sizes, first and warm physical draws, fragment overdraw at representative point diameters, peak RSS, cancellation latency, and visual correctness. The 11.44-MiB logical vertex size alone is not sufficient evidence.

#### Deferred cache simplifications

Removing persisted `ranges/row_start`, quantizing coordinates, adding lazy per-value sidecars, using an uncompressed memory-mapped payload, or offering a value-major-only cache profile each changes a separate contract. Evaluate them only after the dual-ordering prototype has measured results, and keep each in its own schema/benchmark slice.

### Slice 11 — Explicit coordinator selection arming

This slice is a lifecycle and API cleanup rather than a rendering optimization. It makes the product rule explicit: selecting or inspecting a points element may load metadata and available values, but a regular or tiled napari points layer is created only after an explicit Add/Update action.

The current production call path already follows that rule. `TiledPointsController.apply_selection()` is reached from the Add/Update action and passes the requested values to `ViewerAdapter.ensure_tiled_points_layer()` before the adapter constructs the layer and runtime. The current `initial_requested_value_ids` branch does not itself add a layer; it prevents an explicitly created proper-subset layer from briefly issuing an unintended all-values viewport while its first selected-value index is still loading.

The behavior is necessary, but the API is ambiguous:

```python
initial_requested_value_ids: tuple[int, ...] | None = None
```

Here `None` is both the constructor default and the valid semantic representation of an explicit all-values selection. The coordinator therefore cannot distinguish “the application has not configured a selection” from “the user explicitly selected all values.” The constructor also needs special initial-subset flags and failure branches that partly duplicate `set_selected_value_ids()`.

**Production changes**

1. Remove `initial_requested_value_ids` from `_TiledPointsViewportCoordinator.__init__()`.
2. Start the coordinator in an explicit internal `SELECTION_NOT_CONFIGURED` state. Use a private sentinel or selection-state enum; do not use `None` for this state because `None` remains the valid explicit all-values selection.
3. A viewport submitted while selection is not configured may be generation-stamped and retained as the latest desired viewport, but it must not cross into `_TiledPointsCacheSession` or trigger cache reads.
4. Route both first and subsequent selections through one `set_selected_value_ids()` state transition:
   - `None` explicitly arms the coordinator for all values;
   - a nonempty sorted tuple explicitly arms it for a proper subset; and
   - omission is no longer a valid way to select all values.
5. Rename `_TiledPointsLayerRuntime`'s input to required `requested_value_ids` and remove its `= None` default. After constructing the coordinator and connecting listeners, the runtime must explicitly call `set_selected_value_ids(requested_value_ids)` before starting the cache session.
6. Keep `ViewerAdapter.ensure_tiled_points_layer()`'s `requested_value_ids` argument required. It already receives this value only from the explicit controller Add/Update path.
7. Preserve the safe first-subset ordering:
   - retain the latest viewport while the worker commits the selected-value index;
   - dispatch the first viewport only after that commit succeeds; and
   - if the first subset commit fails, report the failure and do not fall back to the session's internal all-values default.
8. Preserve later-update rollback behavior. If a changed selection fails after an earlier explicit selection was committed, keep the earlier committed selection and replan only according to the existing failure policy.
9. Replace the constructor-specific flags such as `_initial_subset_uncommitted` with state derived from explicit desired and committed selections. The coordinator should be able to answer separately whether a selection has been configured, is pending, has been committed, or failed before any commit.
10. Keep layer creation policy outside the coordinator. The coordinator schedules an already explicitly created layer; the controller and adapter remain responsible for deciding whether that layer should exist.

The intended lifecycle becomes:

```text
user selects points element
        -> cache descriptor/value discovery only
        -> no napari points layer

user clicks Add / Update
        -> controller resolves requested_value_ids
        -> adapter constructs layer/runtime
        -> runtime constructs unconfigured coordinator
        -> runtime explicitly arms coordinator with requested_value_ids
        -> cache session starts
        -> first viewport waits for an explicit subset commit when required
```

**Focused tests**

- Selecting/binding a points element and completing descriptor loading does not call `ensure_tiled_points_layer()` or add a napari layer.
- Clicking Add/Update creates or updates exactly one layer through the requested backend.
- A newly constructed, unconfigured coordinator never dispatches a retained viewport when the session becomes ready.
- Explicit pre-start `None` permits the first all-values viewport after readiness.
- Explicit pre-start subset IDs block the first viewport until the selected-value index is committed.
- Initial subset failure never dispatches an all-values viewport.
- A later failed subset change retains the previously committed explicit selection.
- Replacing a pre-start selection retains only the latest explicit selection and generation.
- Runtime construction or startup without an explicit `requested_value_ids` argument fails immediately rather than defaulting to all values.
- Existing-layer Add/Update continues to call the same selection transition without reconstructing the layer.

**Exit condition**

No constructor default can implicitly mean all values, and no viewport cache read can start before the application has explicitly configured the layer's value selection. Metadata discovery remains automatic inside the opt-in panel, while creation of both regular and tiled napari points layers remains an explicit Add/Update action.

### Slice 12 — Conditional identical render-payload reuse

This is an optional follow-up optimization, not part of the required constant-resource renderer or worker-packing milestones. The GPU still redraws the active points on every physical frame; this slice concerns only avoiding redundant CPU packing and VBO replacement when a newly accepted viewport resolves to exactly the same immutable point payload that is already active.

Small pans can change the continuous viewport bounds while retaining the same cache generation, value selection, LOD, and ordered logical tile set. The current complete-snapshot path correctly reuses decoded CPU tiles, but it still constructs a new packed batch and replaces the complete VBO for such an accepted request. That replacement is semantically unnecessary because cache generations and their tile payloads are immutable.

**Entry condition**

Proceed only after Slice 2 is complete and recorded camera traces show that a material fraction of accepted snapshots repeat the active physical payload identity, or that their repeated packing or VBO upload remains a material interaction cost. Do not infer this need from raw camera-event counts: identical viewport states are already suppressed, the coordinator already coalesces active work, and stale snapshots already fail activation.

**Physical payload identity**

Define one canonical, GUI-neutral identity containing:

```text
cache generation
requested value IDs
selected LOD
ordered logical tile keys
```

Request and selection generation counters, viewport bounds, status text, and `omitted_value_ids` are not part of the physical identity. They may change while the vertex rows remain identical and must still be acknowledged and published. Include cache, selection, and LOD explicitly even for an empty ordered tile tuple; an empty tuple by itself is not a sufficient identity.

**Production changes if justified**

1. Add a canonical physical-payload identity to the generation-bound snapshot or render-batch contract. Derive it from already validated fields; do not hash or scan point arrays on the GUI thread.
2. Retain the active physical identity in `VispyTiledPointsLayer`. When an accepted candidate identity equals it, do not call `replace_vertices()` and do not increment the payload-replacement count.
3. Treat reuse as a successful activation. Commit or acknowledge the candidate request and selection generations, clear pending state, and allow composition to publish the candidate status and omission metadata even though the physical VBO did not change.
4. Keep normal full-payload replacement for a changed identity. A changed cache generation, requested-value tuple, LOD, tile membership, or tile order must not take the reuse path.
5. Clear or replace the active identity consistently for an accepted empty snapshot, renderer close, and any transition that suppresses the preceding payload. The layer's cache dataset reference is immutable; switching cache generations constructs a new layer and runtime rather than replacing data in place. A failed candidate must not commit its identity.
6. If worker packing is still material, retain at most one last immutable packed batch, or another explicitly byte-bounded packed-batch cache, keyed by the same identity. Reusing that allocation across snapshots is valid only while it remains read-only and its lifetime across queued Qt delivery is explicit. Account this memory separately from decoded CPU tile residency and GPU bytes.
7. Report accepted snapshot count, physical-identity match count, packed-batch reuse count, VBO replacements avoided, packing time avoided, upload bytes avoided, and retained packed-batch bytes.

The initial implementation should optimize only exact physical-identity matches. Do not include viewport overscan or guard bands, a recent-viewport GPU VBO cache, incremental per-tile VBO mutation, or per-tile visual resources. Overscan changes point-budget and LOD behavior; GPU page retention introduces allocation and draw-range complexity. Either requires separate evidence if exact reuse is insufficient.

**Focused tests**

- Consecutive snapshots with the same physical identity acknowledge the newer request generation without packing again when worker reuse is enabled and without calling `replace_vertices()`.
- Metadata and status changes, including changed omission metadata, are published when geometry is reused.
- Changed cache generation, requested values, LOD, tile membership, or ordered tile identity performs a normal complete replacement.
- Empty-to-empty reuse, nonempty-to-empty activation, and return from empty to a prior nonempty identity preserve correct drawability and replacement counts.
- A stale or failed candidate cannot commit a reusable identity.
- Palette, opacity, and point-diameter changes remain uniform or texture updates and do not invalidate an otherwise reusable vertex payload.
- Any worker-side packed-batch cache remains immutable, byte bounded, generation safe, and allocation-preserving across queued Qt delivery.

**Acceptance evidence**

Replay recorded pan, zoom, resize, and full → partial → full traces. Compare accepted snapshots, worker packs, packed bytes, VBO replacements, upload bytes, first draw after replacement, warm draw, peak RSS, and interaction latency with reuse disabled and enabled. The optimization must materially reduce redundant work without delaying isolated viewport updates or changing visible points, LOD decisions, status, stale-generation rejection, or the one-visual/one-VBO/one-draw topology.

If exact physical-identity matches are rare or their avoided work is not material after the required slices, reject this slice and retain complete snapshot replacement. It is not part of the definition of done.

**Exit condition**

When the evidence gate is met, repeated accepted snapshots with identical immutable geometry reuse the active physical payload while still completing the latest logical activation. Otherwise the measured rejection decision is recorded and no payload cache is added.

### Definition of done for the complete plan

The initial optimization programme is complete when:

1. the default in-memory backend remains unchanged;
2. opt-in tiled rendering uses one visual, one VBO, one worker-prepared batch, and one draw submission;
3. GUI activation contains no tile-proportional packing or resource loop;
4. CPU residency no longer has quadratic no-eviction behavior;
5. proper-subset reads never decode point-level `value_id`;
6. proper-subset Exact reads use a validated value-major coordinate sidecar after LOD selection;
7. all-values and complete-tile requests retain tile-major routing;
8. uncovered levels retain a correct tile-major fallback;
9. bucket sparse ranges are lazy and byte bounded rather than eagerly resident;
10. benchmark reports demonstrate improved cold reads, warm activation, first draw, warm draw, startup RSS, and steady memory on the supplied cache; and
11. the tiled coordinator distinguishes selection-not-configured from an explicit all-values selection, and its first cache read is armed only by the explicit Add/Update path; and
12. debounce, identical render-payload reuse, ping-pong storage, extra sidecar levels, and a larger point budget are accepted only when their own evidence gates are met.

## Conclusion

There are three proven bottlenecks:

1. **Warm and cold rendering: primary problem**

   Thousands of independent VisPy nodes cause a 24-second first draw and approximately one-second warm frames.

2. **Cold point reads: secondary major problem**

   Sparse AAMP ranges trigger 4,291 chunk decodes across 69 serial bucket batches, taking approximately 4.14 seconds.

3. **Quadratic residency bookkeeping: avoidable implementation defects**

   CPU retention adds approximately 852 ms; GPU retention adds approximately 1.84 seconds.

The practical priority is therefore:

1. Replace one-visual-per-logical-tile rendering with one snapshot visual/program and one VBO fed by worker-prepared immutable render batches. This removes the quadratic GPU residency path rather than optimizing a tile-resource design that is no longer needed. Preserve logical tiles only at the storage and CPU-residency boundaries, and keep tile-proportional packing off the GUI thread. Add a second VBO only if measured behavior justifies ping-pong storage.
2. Fix the quadratic CPU residency path, which remains useful for reusing decoded logical tiles across viewport requests.
3. Stop reading point-level `value_id` for proper selected-value requests; construct it from the requested values and resolved catalog or fallback intervals, or represent a one-value snapshot with a uniform.
4. Prototype an Exact-level-only, coordinate-only value-major sidecar alongside the existing tile-major payload. This deliberately duplicates the Exact coordinate bytes, projected at approximately 0.79 GiB and a 50% cache increase, while reusing the manifest and value-to-tile catalog and omitting duplicate point-level `value_id` and `point_id` arrays. Choose LOD first, then route all-values and complete-tile reads to tile-major, proper-subset reads to a sidecar available at the chosen level, and uncovered proper-subset levels to tile-major fallback. Replace eager retention of the complete 568.4 MiB sparse bucket lookup with compact always-resident addressing plus lazy, byte-bounded fallback indexes. Measure actual compressed size, cold and warm selected-value wall time, decoded bytes, physical operations, startup and peak lookup memory, and fallback churn. Extend the sidecar to other levels only if runtime evidence justifies their additional storage.
5. Treat smaller chunks, fewer buckets, and cross-bucket concurrency as secondary comparisons or tuning. The current evidence does not support them as fixes for tile-major sparse decoding or per-tile rendering.
6. Add viewport debounce to avoid starting expensive cold requests for transient zoom states after the underlying read and render costs are controlled.
7. Add exact render-payload reuse only if recorded camera traces show that accepted viewports frequently resolve to the active immutable tile identity and that avoiding their packing or upload is material.

The synthetic 1,000,000-point packing benchmark does not change the current 100,000-point implementation priority. It establishes forward-looking scalability evidence and the additional acceptance work required before the render budget is deliberately increased.

Larger storage tiles would reduce chunks and visual objects, but the benchmark shows that renderer batching can remove the dominant visual-object cost without sacrificing the existing 512-unit spatial read granularity.

This implementation-planning update changes only this roadmap document; it does not implement the slices above.
