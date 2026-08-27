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
2. Record `pending_keys` for logical diagnostics only; tile keys are no longer GPU resource identities.
3. Validate that the worker-prepared render batch is immutable, C-contiguous, reconciles with `rendered_point_count`, and has the expected 12-byte vertex format.
4. Preflight that the new payload fits the configured single-buffer byte budget before mutating the VBO.
5. Replace the sole VBO payload with the prepared array using the selected copy-safe lifetime semantics.
6. If `set_data()` raises a Python exception, emit the render error, clear pending state, and do not commit the candidate generation or keys. The previous GPU payload is not guaranteed to remain drawable after mutation has begun.
7. On success, update the point count, set visual drawability according to whether the snapshot is empty, and commit `active_keys` plus the accepted result.
8. Request one scene update.

Packing failures now occur on the worker before a candidate snapshot is submitted. They should follow the existing worker-error path and leave the active visual untouched. `apply_snapshot()` must not contain the snapshot tile loop in the completed implementation.

This deliberately changes overlap behavior: a changed accepted snapshot performs one bounded upload even when it shares tiles with the preceding snapshot. Warm CPU tile reuse still avoids Zarr reads and decoding. The renderer trades per-tile GPU reuse for a constant scene graph and a single draw submission; at the 100,000-point cap that is the correct trade.

Validation, cancellation, stale-generation rejection, over-budget rejection, and byte-capacity rejection all occur before `set_data()` and therefore continue to preserve the active payload. The initial one-VBO design deliberately does not promise rollback after VBO replacement starts. VisPy defers the actual GPU operation until rendering, so even a ping-pong design would require explicit draw-error handling before it could claim verified GPU-upload rollback.

`_VispyTileResource`, `_GpuTileResidency`, per-tile visibility loops, per-tile LRU retention, and per-tile GPU eviction cease to be part of the normal renderer. `active_keys` and `pending_keys` may remain because they are useful for request identity and diagnostics, but they must not own resources. Existing metrics such as resident GPU tile count and GPU eviction count should be replaced with visual count, VBO count, active point count, active bytes, candidate batch bytes, pack time, and upload-staging time. A temporary compatibility alias is acceptable if another internal consumer still reads an old field.

#### Budget implications

With the existing 12-byte vertex format:

```text
100,000 points * 12 bytes = 1,200,000 bytes = approximately 1.15 MiB for the VBO
optional two-VBO ping-pong maximum = approximately 2.29 MiB
1,000,000 points * 12 bytes = 12,000,000 bytes = approximately 11.44 MiB for one VBO or packed batch
optional two-VBO ping-pong maximum at 1,000,000 points = approximately 22.89 MiB
```

The initial patch can retain the existing GPU-byte configuration name to limit configuration churn, but its enforcement should cover the one snapshot payload rather than a sum of tile estimates. The bounded worker batch and any temporary CPU copy made by VisPy should be reported separately from logical GPU bytes. At 1,000,000 points, the worker output is approximately 11.44 MiB, the VBO is approximately 11.44 MiB, and `set_data(copy=True)` may temporarily retain another approximately 11.44 MiB CPU staging copy. A whole-snapshot concatenate/repeat implementation also uses temporary coordinate, value-ID, and repeated-origin arrays; its peak must be measured rather than inferred from the final VBO size alone.

A later cleanup can rename the setting to a snapshot-buffer budget or derive it directly from the hard point budget. Both the 100,000- and 1,000,000-point final VBOs are far below the current 512 MiB default, but an increase to 1,000,000 points still requires deliberate end-to-end memory, upload, and draw evidence rather than only a logical byte calculation.

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

The synthetic 1,000,000-point packing benchmark does not change the current 100,000-point implementation priority. It establishes forward-looking scalability evidence and the additional acceptance work required before the render budget is deliberately increased.

Larger storage tiles would reduce chunks and visual objects, but the benchmark shows that renderer batching can remove the dominant visual-object cost without sacrificing the existing 512-unit spatial read granularity.

This follow-up investigation changed only this roadmap document. No repository source code was changed.
