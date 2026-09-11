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

The strongest cache-side solution is to retain the existing tile-major payload and add a display-only, per-level location payload ordered by:

```text
(value_id, manifest_index, point_id)
```

The two physical representations serve different access patterns:

| Access pattern | Physical payload |
|---|---|
| All values or complete logical tiles | Existing tile-major bucket payload |
| Proper value subset at any selected level | Mandatory value-major location payload for that level |

#### What is physically duplicated

This is genuine physical duplication of the location rows. A Zarr array has one physical row order, so the same logical locations must be materialized once in tile-major order and once in value-major order. An alternate index into the existing tile-major rows would not solve the decode problem: the index could find AAMP's rows, but those rows would still be scattered across the same tile-major chunks.

It is not necessary to duplicate the complete cache or every per-point field:

| Existing structure | Value-major sidecar | Reason |
|---|---|---|
| `location` | Duplicate and reorder | These are the bytes that must become contiguous for selected-value reads. |
| `value_id` | Omit | The requested value and its catalog interval already identify every returned row. |
| `point_id` | Omit | It can establish deterministic construction order and then be discarded from the display sidecar. |
| tile manifest and transforms | Reuse | `manifest_index` still identifies the tile offset for tile-relative coordinates. |
| value-to-tile catalog | Reuse | The catalog already identifies each value's positive tiles and point counts. |
| bucket sparse-range metadata | Do not duplicate | Retain it initially for construction, catalog generation, and validation; the viewer-side value-major path does not consume it. |

A minimal representation is therefore conceptually:

```text
value_major/
    level_0/
        location              # float32 [N_exact, 2]
        value_point_indptr    # compact start/stop per canonical value
    level_1/
        location              # float32 [N_bridge, 2]
        value_point_indptr
    ...
    level_N/
        location              # float32 [N_level_N, 2]
        value_point_indptr
```

Within every serialized level, rows in `location` are ordered by `(value_id, manifest_index, point_id)`, but only the locations are persisted. That level's `value_point_indptr` gives each canonical value's complete location interval. It is compact because it has one pointer per level/value rather than one pointer per value/tile record.

The cache catalog already persists value-to-tile records in `(level, value_id, manifest_index)` order. The sidecar follows that same record order. The current selected-value index retains the aligned `manifest_index` and `n_points` records only for the active selection. A cumulative sum of those selected counts derives per-record location offsets in memory; a cache-wide persisted or resident `record_point_indptr` is therefore unnecessary. A full-extent AAMP read becomes one contiguous value interval; a rectangular partial viewport becomes a small set of value-major spatial runs rather than one interval in each positive tile. The returned tile-relative locations can still be split by catalog record and combined with the existing manifest tile offsets by the snapshot packer.

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

These are estimates, not rebuilt-cache measurements. Changing row order can improve or worsen compression, so actual compressed bytes must be recorded by the first rebuilt cache. The mandatory all-level sidecar duplicates approximately 1.09 GiB of location rows, not the full 1.57 GiB cache and not `point_id` or `value_id`.

The supplied cache's current compressed tile-major `location` payload is distributed as approximately 814.3 MiB for Exact, 129.5 MiB for Bridge, and 176.9 MiB for all Spatial levels combined. Relative to the 943.8-MiB Exact-plus-Bridge location portion, covering every remaining Spatial level therefore adds only approximately 177 MiB of current location payload, subject to remeasurement after value-major reordering. This relatively small incremental cost buys one proper-subset physical route at every LOD.

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

The initial dual-ordering cache should preserve the existing bucket sparse ranges on disk for cache construction, catalog generation, and independent publication validation. Viewer rendering no longer needs them: every proper-subset level has a sidecar, while all-values tile-major reads need only complete-tile addressing and point-level `value_id`.

They should no longer be one indivisible, eagerly resident startup index. The runtime policy should be:

| Index data | Residency policy |
|---|---|
| manifest, value pointers, value totals | Always resident |
| active selected value-to-tile records and counts | Resident for the committed selection |
| per-level `value_point_indptr` arrays for sidecar addressing | Always resident; compact |
| tile-major `tile_offset` | Always resident or derived once; compact |
| bucket `ranges/{tile_indptr,value_id,row_start,row_count}` | Persist for construction and validation; do not load into the viewer runtime |

A value-major request must not load the bucket sparse-range arrays. An all-values tile-major request needs only the complete tile interval and point-level `value_id`; it also does not need the sparse-range arrays. Consequently, no normal viewer request needs a sparse-range lookup index and the runtime does not need an LRU or fallback-index byte budget.

Keep `ranges/row_start` on disk for the first sidecar slice to avoid combining the all-level physical-order rewrite with construction and validation changes. It is a candidate for a later schema simplification because validated ranges partition each tile contiguously: their starts can be reconstructed from `tile_offset` plus a cumulative sum of `ranges/row_count` within each tile. Removing it requires explicit size, construction-memory, and validation evidence. Optional Slice 17 evaluates removal of the complete persisted bucket sparse-range group, with `row_start`-only removal as a smaller alternative.

Point-level `bucket/value_id` is distinct from `ranges/value_id`. Keep the point-level array in the tile-major payload because all-values rendering needs a colour ID aligned with every coordinate. Proper-subset value-major reads construct the output IDs from known value intervals instead of decoding that point-level array. Deferred Slice 11's complete-tile filtering route would instead need the stored point-level IDs to remove unselected rows, as the independent diagnostic/reference filtering path already does.

#### Explicit physical-payload routing

LOD selection must happen before physical-payload selection. The semantic level still comes from the viewport, selected values, and point budget; the existence of a sidecar must not force Exact when the request requires a coarser level.

After the level is selected, use this deterministic initial routing rule:

| Request after LOD selection | Physical payload | Large bucket sparse-range index |
|---|---|---|
| Over budget | Read neither payload | Not needed |
| All canonical values | Tile-major at the selected level | Not needed |
| Complete-tile or construction access | Tile-major | Not needed for complete row access; publication validation remains separate |
| Proper value subset at any selected level | Mandatory value-major sidecar for that level | Not needed |

Selecting the complete vocabulary is already normalized to the all-values state, so it follows the tile-major branch. For the supplied full-extent, 100,000-point case, AAMP selects Exact with 60,512 points and therefore uses the Exact value-major sidecar; the all-values request selects Spatial level 8 with 100,000 points and therefore uses that level's tile-major payload.

Throughout the current interaction milestone, including Slices 12 and 13, every proper subset uses the mandatory sidecar belonging to the semantically selected level. This rule is reproducible, removes LOD-dependent fallback behavior, and prevents sparse tile-major range decoding from returning merely because a dataset selects Bridge or Spatial. Deferred Slice 11 later adds a measured physical cost comparison for proper subsets: a dense, near-all-values subset in a small viewport may be cheaper to read as complete tile-major tiles and filter in memory, while a sparse value spread across many tiles may remain cheaper through value-major. That route must not depend on the viewer loading the legacy sparse-range indexes. It must compare projected touched chunks or shards, decoded rows or bytes, and physical operations rather than using only the number or fraction of selected genes. The decision belongs to the cache reader after LOD selection and CPU-residency lookup, not the GUI or renderer, and is made once for the complete missing-tile request so both routes produce the same logical tile payload and reuse the same CPU-residency contract.

All-level coverage guarantees that a locality-oriented payload exists at every LOD; it does not guarantee constant-time rendering for every future selection. A proper subset containing many disjoint values can still touch many value-major intervals, a partial viewport can still require several spatial runs, and decoding, packing, upload, and drawing remain real bounded costs. The point budget limits returned rows, while the integrated benchmark must verify physical amplification and interaction latency rather than treating sidecar presence alone as sufficient evidence.

#### Recommended staged implementation

The first cache-side implementation should retain a deliberately narrow payload while covering every LOD:

1. Always build a **location-only value-major sidecar for every serialized level** in `(value_id, manifest_index, point_id)` order as part of the cache format.
2. Reuse the existing manifest and value-to-tile catalog, persist only compact per-value coordinate pointers, and derive selected per-record offsets from catalog counts.
3. Apply the post-LOD routing table above throughout the interaction milestone: proper-subset reads use the selected level's sidecar, while all-values and complete-tile reads use tile-major. Implement stable coverage and establish its integrated baseline before revisiting deferred Slice 11, which may route a proper subset to complete tile-major reads plus in-memory filtering when its measured physical cost model predicts that route is cheaper.
4. Remove the current eager bucket sparse-range lookup policy from the viewer runtime; do not replace it with a fallback-index cache.
5. Measure total construction time and per-level compressed size together with cold and warm selected-value reads, decoded bytes, physical operations, startup, and peak lookup memory for sparse and dense values at full and partial viewports.

Cache construction time is intentionally not an acceptance constraint unless it becomes operationally prohibitive. Measurements must show how much interaction improves for the extra approximately 1.09 GiB and may guide physical-layout tuning, but they do not make any serialized level's sidecar optional in newly built caches.

If that storage increase later proves operationally unacceptable, lower-storage variants can be evaluated as explicit future schema redesigns, not as a switch that omits the mandatory sidecar from the current schema:

- build persistent value-major locations lazily for selected or frequently used values; AAMP's 60,512 float32 two-dimensional locations are only approximately 0.46 MiB raw, excluding metadata;
- store explicitly validated, display-only quantized tile-relative coordinates, for example `uint16`, which halves the raw coordinate width but introduces a precision contract; or
- offer a value-major-only cache profile for workflows that do not require efficient all-value or complete-tile reads, accepting the loss of the current primary access order.

This design deliberately spends cache-construction time and storage to improve runtime behavior. It also creates a path that does not require the complete 568.4 MiB bucket sparse-range lookup to be resident for selected-value reads. The existing small selected-value catalog index can perform discovery, while the value-major payload provides direct coordinate access.

### Complementary cache improvements and comparison baselines

One smaller runtime change remains independently justified:

1. **Do not read point-level `value_id` for proper subsets.**

   `resolve_selected_tile_intervals()` already knows the selected value associated with each range. Reconstructing the aligned IDs from the resolved ranges would remove the measured 1.86-second `value_id` Zarr boundary for AAMP. A one-value renderer could alternatively use a uniform value ID.

   This applies to the range-resolved path and, later, to the value-major path whose pointer intervals also imply each value ID. Deferred Slice 11 defines a future production exception: a proper subset may read complete tile-major `value_id` rows when it chooses `tile_major_filter`, because those IDs are then required for in-memory membership filtering and the measured total physical cost is lower. That adaptive route is not part of the current interaction milestone; independent diagnostic/reference filtering remains separate from viewer routing.

The existing smaller-chunk and fewer-bucket benchmarks did not materially solve the end-to-end problem. A 128- or 256-row `location` setting may be retained as a controlled comparison when measuring the first value-major build, but it is not a recommended implementation slice on its own. Any further comparison must include cache size, inner-chunk index size, physical reads, and wall time; decoded-row reduction alone is not sufficient acceptance evidence.

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
        -> discard transient decoded-tile references not retained by CPU residency
        -> TiledPointsRenderSnapshot(rendered_tile_count=..., render_batch=...)
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
| `contracts.py` | Carries a tuple of logical render tiles in `TiledPointsRenderSnapshot`. | Defines and validates an immutable, C-contiguous packed render batch and carries it with the generation-bound snapshot. The snapshot retains only the O(1) logical-tile count; decoded tiles remain worker-local. |
| `runtime/cache_session.py` | Assembles the ordered logical tile tuple. | Validates and packs the render batch after final ordered tile assembly, records `rendered_tile_count`, and returns no decoded tile arrays across the GUI boundary. |
| `runtime/composition.py` | Delivers the snapshot to the layer event. | Forwards the snapshot and packed batch unchanged; it remains unaware of VisPy and VBO ownership. |

#### Snapshot packing

Add one pure NumPy packing helper in a GUI-neutral module such as `viewer/tiled_points/render_batch.py`. It must not live under `viewer/tiled_points/vispy/`: importing that package imports the VisPy layer and its GUI dependencies, which the cache worker should not acquire. The helper should:

1. Allocate one structured array sized exactly to the worker's declared `point_count`.
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

The runtime should therefore record both point count and tile count for every packed snapshot. A hard tile-count limit should not be introduced from this synthetic result alone because sparse selected values may legitimately span many tiles. Instrumentation should first establish a point-plus-tile work model that can inform LOD planning or a future preparation-work budget. Slice 2 keeps packing cooperatively cancellable through the cache session's existing terminal-close event. It does not add a second request-specific cancellation protocol: the existing one-active/one-latest-pending coordinator may let an active obsolete request finish, rejects its completed generation before renderer submission, and then dispatches the latest request. Add per-request packing cancellation only if profiling shows that obsolete, highly fragmented packs materially delay the latest request; that follow-up requires an explicit thread-safe request-cancellation token and tests distinct from session closure.

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

`_VispyTileResource`, `_GpuTileResidency`, per-tile visibility loops, per-tile LRU retention, per-tile GPU eviction, and renderer-owned active or pending tile-key tuples cease to be part of the normal renderer. The snapshot and runtime already carry generation-bound logical identity; the renderer needs only the active point count and payload metrics. Slice 12 introduces a dedicated worker-prepared physical-payload identity for coverage reuse rather than reconstructing tile-key identity on the GUI thread. Existing metrics such as resident GPU tile count and GPU eviction count should be replaced with visual count, VBO count, active point count, active bytes, candidate batch bytes, pack time, and upload-staging time. A temporary compatibility alias is acceptable if another internal consumer still reads an old field.

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

Slice numbers remain stable identifiers, not a requirement to implement them in numerical order. The next implementation is Slice 12, followed by Slice 13 on the existing physical routes. Slice 11 is deferred to a later loading-optimization phase.

| Slice | Primary result | Depends on | Status after merge |
|---|---|---|---|
| 0 | Preserve the opt-in boundary and freeze the baseline | Current code | Completed starting point |
| 1 | Replace per-tile GPU resources with one visual and one VBO | Slice 0 | Dominant draw-submission problem removed; temporary GUI packing allowed |
| 2 | Move complete render-batch packing to the worker | Slice 1 | Completed smooth-frame renderer architecture |
| 3 | Make CPU tile retention linear for the no-eviction case | Slice 0 | Cold CPU assembly defect removed |
| 4 | Enforce homogeneous bucket display batches | Slice 0 | Mixed all-values/subset batches fail before planning or physical IO |
| 5 | Stop decoding point-level `value_id` for proper subsets | Slice 4 | One of two selected-value Zarr reads removed through one explicit batch mode |
| 6 | Make a location value-major sidecar mandatory at every serialized level in the new cache format and writer | Slice 0 | Every newly built cache contains every sidecar and validates its structural and index contract, but the viewer does not use it yet |
| 7 | Add optional exhaustive value-major location-equivalence validation | Slice 6 | Developer-only proof that sidecar locations equal tile-major locations; not a publication dependency |
| 8 | Initially route every proper-subset read through the selected level's sidecar | Slices 5 and 6 | Selected values gain locality at every LOD |
| 9 | Remove duplicated per-tile selected-value membership from viewport plans | Slice 8 | One authoritative value-to-tile relation and a leaner semantic plan |
| 10 | Remove bucket sparse-range indexes from the viewer runtime | Slices 8 and 9 | Startup time, lookup RSS, and fallback-cache complexity are removed |
| 10b | Make complete-tile addressing self-contained in `_TileDescriptor` | Slice 10 | Required tile index and point-row start replace the separate `_BucketTileIndex`, preserving first-use validation |
| 10c | Remove the complete-tile fallback and retire the resident bucket sparse lookup | Slice 10b | One complete-tile display contract; diagnostic subsets filter tile-major point arrays without changing the cache format |
| 10d | Consolidate value-major array ownership in a per-level reader | Slice 10c | One `_ValueMajorLevelReader` per level owns both Zarr array references, validates their layouts, loads pointers, and performs bounded location reads |
| 11 | Deferred: add measured adaptive routing for proper subsets | Slices 9, 10b, 10c, 10d, 12, and the initial Slice 13 baseline | Later dense-subset loading optimization calibrated on actual coverage misses; not an interaction prerequisite |
| 12 | Next: add stable render coverage, LOD hysteresis, and bounded packed-batch reuse | Slices 2 and 8–10, including 10b, 10c, and 10d; not Slice 11 | Smooth interaction on both sides of the 100,000-point boundary using the existing physical routes |
| 13 | Run the integrated all-level acceptance and tuning matrix | Slices 1–6, 8–10, 10b, 10c, 10d, and 12; Slice 7 optional; not Slice 11 | Evidence-backed interaction baseline with fixed routing; adaptive crossover measurements deferred |
| 14 | Add viewport debounce only if dispatch churn remains material | Slice 13 | Conditional reduction of obsolete work after coverage reuse |
| 15 | Evaluate optional ping-pong storage and a larger point budget | Slice 13 | Conditional hardening/scaling work, not part of the initial solution |
| 16 | Replace implicit initial selection with explicit coordinator arming | Slice 0 | No unconfigured or accidental all-values first viewport |
| 17 | Evaluate optional removal of persisted bucket sparse ranges | Slice 13 | Conditional schema cleanup with construction-only range records and adapted validation/reference consumers |

Slices 1 and 2 form one renderer milestone. Slice 1 may be reviewed and measured independently, but Slice 2 is required before the renderer work is considered complete. Slices 6 and 8 form one cache-locality milestone: publishing all-level sidecars that no read path consumes is useful only as a short-lived, testable construction boundary. Slice 7 is an optional developer-validation layer between those production slices and is not a prerequisite for publication or runtime routing. Slice 9 removes the duplicate per-tile projection introduced by the first sidecar reader. Slices 8 through 10 establish the simple sidecar-first runtime without sparse indexes; Slices 10b through 10d simplify its reader contracts.

Prioritize Slice 12 now because recorded warm requests still spend substantial time planning and packing with zero Zarr payload reads. Stable coverage and packed-batch reuse address that repeated work; choosing a different storage route cannot. Slice 12 keeps proper subsets on value-major and all-values requests on tile-major, then Slice 13 measures both cold coverage construction and warm interaction with those fixed routes. Keep cold/dense-subset limitations visible in that baseline rather than making adaptive routing a prerequisite for measuring the stutter improvement.

Slice 16 is an independent lifecycle cleanup that can follow the initial Slice 13 baseline; rerun the affected interaction/lifecycle checks after it lands. Evaluate Slice 14 only for remaining obsolete dispatch churn, and Slice 15 only for demonstrated GPU-update hardening or scaling needs. Optional Slice 17 separately evaluates persisted-range removal after the baseline and is lower priority for stutter because the viewer no longer loads those indexes. Slices 14, 15, and 17 may be rejected or deferred at their evidence gates; implementing all three is not required to complete the interaction milestone or to revisit Slice 11.

Revisit Slice 11 in the later loading-optimization phase using the remaining coverage-miss workload established by Slices 12 and 13. The existing `read_planned_tiles(plan, tile_keys_to_read)` boundary already accepts explicit misses, so neither coverage identity nor CPU-residency keys need a physical-route dependency. Defer the estimator, production `tile_major_filter`, and forced adaptive-route acceptance checks together. Calibrating them after coverage reuse measures the requests that actually reach storage rather than the superseded exact-viewport workload.

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

This is a renderer-ownership change, not a cache-layout or logical-tile change. At the Slice 1 boundary, the worker still returns a `TiledPointsRenderSnapshot` containing the complete ordered tuple of logical `TiledPointsRenderTile` objects for the accepted viewport. CPU tile residency also remains unchanged. Slice 2 subsequently keeps those decoded tiles worker-local and returns only their count plus the packed batch.

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

The accepted viewport may still comprise 4,453 logical tiles, but those tiles are no longer GPU ownership units. Every accepted nonempty snapshot is packed into one complete vertex payload and replaces the contents of the same stable VBO. Overlapping logical tiles between successive viewports are therefore not reused as independent GPU buffers; full-payload replacement is deliberate because it gives constant visual, VBO, and draw-submission counts. The renderer does not retain or reconstruct active or pending tile-key tuples. Slice 12 later adds worker-prepared coverage identity and bounded packed-batch reuse without restoring per-tile GPU ownership.

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
6. Do not retain or reconstruct active or pending tile-key tuples in the renderer. Slice 12 adds its dedicated worker-prepared physical-payload identity at the coverage boundary.
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

**Status: Implemented**

This slice completes the renderer milestone by removing the tile loop from GUI activation.

**Production changes**

1. Refactor the Slice 1 `pack_snapshot_vertices()` scaffold into a `pack_render_tiles()` helper in `viewer/tiled_points/render_batch.py`. The helper must accept the complete ordered logical tile tuple plus explicit `point_count`, `value_count`, `max_vertex_payload_bytes`, and an optional cancellation callback; it must not accept a fully constructed `TiledPointsRenderSnapshot`. This avoids a construction cycle once the final snapshot is required to contain the batch. The helper:
   - allocates exactly one primary structured array;
   - preflights `point_count * TILED_POINTS_VERTEX_DTYPE.itemsize` against `max_vertex_payload_bytes` before allocation;
   - fills consecutive slices in the supplied final tile order;
   - folds `(tile_x * tile_size, tile_y * tile_size)` into `float32` positions;
   - writes canonical value IDs as exactly representable `float32` values;
   - validates the declared point count, final cursor, and maximum palette ID;
   - invokes the optional cancellation callback before allocation and periodically between tile groups; and
   - returns a C-contiguous, owning, read-only allocation, including a valid empty batch.
2. In `viewer/tiled_points/contracts.py`, move the canonical `TILED_POINTS_VERTEX_DTYPE` beside a new immutable `TiledPointsRenderBatch` contract so both validation and packing depend on one definition without making `contracts.py` import `render_batch.py`. Carry the batch on `TiledPointsRenderSnapshot`. Validate dtype, one-dimensional shape, ownership, C contiguity, read-only state, and byte count. Expose O(1) batch `point_count` and `nbytes` properties. Replace the decoded `tiles` tuple on the GUI-bound snapshot with a validated nonnegative `rendered_tile_count`. For a within-budget snapshot, require that the batch count equals `estimated_point_count` and that the tile count is possible for the nonempty logical-tile contract; an over-budget metadata-only snapshot carries zero rendered tiles and an owning read-only empty batch even when its estimate is nonzero.
3. Make `TiledPointsRenderSnapshot.rendered_point_count` return the validated render-batch point count in O(1). GUI-side status preparation, renderer activation, and diagnostics obtain both point count and logical-tile count without inspecting decoded tiles.
4. In `runtime/cache_session.py`, first restore `ordered_tiles = tuple(payloads_by_key[key] for key in keys)` in final plan order. Validate tile-key uniqueness, spatial order, cache generation, selection and level while the tuple is still worker-local; validate the declared point count and byte capacity while packing; check cancellation again; and only then construct the final snapshot from `len(ordered_tiles)` and the immutable batch. For an over-budget result, construct the metadata-only snapshot with zero rendered tiles and the canonical empty batch, and perform no point-payload allocation.
5. The cancellation callback used by Slice 2 is the existing cache-session terminal-close check. Check it before packing, between fragmented tile groups, and after packing so layer/session closure cannot publish a late batch. Do not add request-specific cancellation in this slice. Obsolete request generations may finish packing but must continue to be rejected by the coordinator before renderer submission.
6. Keep decoded logical tiles only in worker-local assembly and `_CpuTileResidency`. Do not transport their coordinate/value arrays in `TiledPointsRenderSnapshot`: after packing, release transient tiles that were not retained by the byte-bounded residency. Carry only `rendered_tile_count` for status and diagnostics. The renderer consumes only `render_batch`.
7. Keep `runtime/composition.py` transport-only: it forwards the generation-bound snapshot and batch without knowing the vertex format or VisPy ownership. Its status path consumes only O(1) snapshot counts.
8. In `vispy/layer.py`, remove the scaffold packing call and renderer-owned pack timing. GUI activation validates the already prepared batch, independently preflights its point and byte capacity, stages exactly that one VBO payload, acknowledges the result, and updates the scene. It performs no logical-tile iteration or NumPy coordinate packing.
9. Initially use the copy-safe `VertexBuffer.set_data()` lifetime behavior already relied on by the renderer and report any CPU staging copy separately. A later zero-copy change requires explicit lifetime and deferred-upload evidence.
10. Rename `max_gpu_tile_bytes` to `max_vertex_payload_bytes` throughout `TiledPointsApplicationSettings`, `TiledPointsLayerModel`, application-adapter construction, renderer capacity validation, diagnostics, benchmarks, and tests. Remove the old name outright: do not add a deprecated constructor keyword, property, configuration alias, or fallback. Add `max_vertex_payload_bytes` to `_CacheSessionSettings` and pass it to the worker because the primary allocation now happens there. The worker preflights the declared batch size before allocation, and the renderer repeats the validation defensively before VBO staging. Continue to enforce the renamed setting against the logical byte size of one complete packed vertex payload, separately from the hard point-count budget and from measured transient memory.

**Focused tests**

- Add direct packing tests for multiple tiles, sparse tiles, ordering, offsets, values, empty batches, large cache-relative positions, incorrect declared counts, and palette overflow.
- Assert the packed batch owns one read-only C-contiguous allocation and uses the expected 12 bytes per point.
- Add cache-session tests proving that resident and newly read tiles are packed in final plan order.
- Add terminal session-close cancellation tests before and during a highly fragmented pack; do not represent these as request-generation cancellation tests.
- Assert that obsolete request generations never submit their completed batch to the renderer.
- Add a queued Qt-delivery test for a maximum-budget batch and verify that the vertex allocation identity is preserved across signal delivery.
- Instrument GUI-side status preparation and `apply_snapshot()` in tests and assert that neither performs tile iteration or NumPy coordinate packing.
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

**Implemented evidence (2026-08-29)**

The full-extent AAMP run used 60,512 points across 4,453 logical tiles and produced one immutable 726,144-byte batch. After the decoded-tile tuple was removed from the GUI-bound snapshot, cold and warm worker packing measured 20.4 and 19.9 milliseconds. Worker-local tile/key validation measured 1.08 and 1.09 milliseconds, while construction of the final metadata-plus-batch snapshot envelope measured 0.030 and 0.017 milliseconds. Queued Qt delivery had a 0.118-millisecond median over five repeats and preserved the exact NumPy vertex-allocation identity on every delivery.

The real-canvas rerun retained exactly one visual, one VBO, and one point draw submission. GUI `apply_snapshot()` measured 0.52 milliseconds, including 0.49 milliseconds of synchronous `VertexBuffer.set_data(copy=True)` staging. The first physical draw measured 16.71 milliseconds and the five-repeat warm-draw median was 3.65 milliseconds. Replacing the full payload again drew in 3.60 milliseconds, confirming that the first-draw result includes one-time pipeline work. The report is `/private/tmp/napari-harpy-slice2-cache-to-canvas-full-aamp.json`.

The five-repeat synthetic 1,000,000-point packing medians were:

| Logical tiles | Immutable worker-batch packing median |
|---:|---:|
| 1 | 6.92 ms |
| 4,453 | 25.28 ms |
| 7,294 | 36.84 ms |
| 100,000 | 415.35 ms |

Every synthetic batch contained 12,000,000 bytes using the canonical 12-byte vertex format. These measurements are slower than the earlier isolated scaffold measurements because they cover the production helper's input-contract validation and immutable batch construction as well as the coordinate-copy loop. They preserve the same conclusion: point count alone remains manageable, while extreme logical-tile fragmentation is the important worker-latency risk. The report is `/private/tmp/napari-harpy-slice2-synthetic-million-point.json`.

Focused validation passed with 141 tiled-points tests, the opt-in native OpenGL qualification, and all 95 `test_viewer_widget.py` backend tests. The worker-session coverage includes final plan order and rejection of nonspatial order before packing, worker-local decoded-tile ownership, pre-allocation byte rejection, terminal-close cancellation before and during packing, and queued-delivery allocation identity. GUI composition coverage reads selected value IDs from the packed batch and no longer receives decoded tile arrays.

**Exit condition**

The completed path is ordered worker-local tiles → tile/key validation → byte preflight → immutable packed batch → discard nonresident decoded-tile references → validated metadata-plus-batch snapshot construction → queued reference delivery → one GUI-thread VBO replacement → one draw. GUI activation obtains point, tile and byte counts in O(1), no decoded tile arrays cross the GUI boundary, and no normal renderer code iterates or creates a resource per logical tile. `max_vertex_payload_bytes` is enforced before the worker allocation and again before VBO staging, is the sole production setting name for the complete packed-payload byte bound, and has no `max_gpu_tile_bytes` input alias or property.

### Slice 3 — Linear CPU tile retention

**Status: Implemented**

This slice removes the measured CPU-residency defect without changing residency semantics. The latest full-extent AAMP rerun retained 4,453 decoded tiles containing approximately 0.69 MiB of point arrays but spent 846.93 milliseconds in the one `_CpuTileResidency.retain()` call, consistent with the earlier approximately 852-millisecond result.

**Pre-implementation defect**

After viewport planning, `runtime/cache_session.py` looks up every planned key in `_CpuTileResidency`, reads and detaches misses, assembles the complete worker-local ordered tile tuple, and calls `residency.retain(new_tiles, protected_keys=resident_keys)` before render-batch packing. `get()` promotes hits to most-recently-used order; `protected_keys` prevents the resident tiles used by the current candidate viewport from being evicted while its newly decoded misses are considered.

Before this slice, `retain()` called `_evict_until_fits()` once for every eligible new tile. `_evict_until_fits()` began with:

```python
for key in tuple(self._entries):
    if self._resident_bytes + required_bytes <= self._max_resident_bytes:
        return
```

Python materialized `tuple(self._entries)` before the first capacity check. Therefore, even when every tile fit and no eviction occurred, insertion 1 copied zero existing keys, insertion 2 copied one, and insertion 4,453 copied 4,452. The complete no-eviction operation copied approximately 9.9 million key references and scaled as `O(N²)` rather than with the number of inserted tiles.

**Production changes**

1. In `runtime/residency.py`, give `_evict_until_fits()` a constant-time capacity fast path before creating an iterator or temporary key/victim collection:

   ```python
   if self._resident_bytes + required_bytes <= self._max_resident_bytes:
       return
   ```

2. Only when eviction is required, calculate the missing capacity and traverse the `OrderedDict` lazily from least to most recently used. Skip protected keys, collect only the oldest unprotected victims needed to release sufficient bytes, and stop as soon as that total is reached. Delete the collected victims after traversal; do not mutate the `OrderedDict` during direct iteration and do not materialize all resident keys up front.
3. Keep the existing `_require_consistent_bytes()` reconciliation outside the per-tile insertion loop. One complete-operation `O(N)` accounting check remains acceptable; production must not perform a complete residency scan after each inserted tile.
4. Preserve the current retention contract exactly:
   - `get()` promotes a hit to most recently used;
   - the oldest unprotected entries are evicted first;
   - keys used by the current candidate viewport remain protected;
   - entries and incoming tiles retain deterministic caller/LRU order;
   - replacing an already resident key updates accounting and moves the successful replacement to most recently used order;
   - a previous value is restored if its replacement cannot fit;
   - a tile larger than the complete budget is never retained; and
   - a tile blocked by protected capacity remains a transient worker-local payload.
5. Do not change viewport assembly or rendering to accommodate a transient tile. The local `ordered_tiles` tuple still owns it through packing, so it appears in the current `TiledPointsRenderBatch`; only its reuse by a future viewport is forfeited.

**Focused tests**

- Extend `runtime/test_residency.py` with no-eviction bulk retention, protected eviction, insufficient-capacity, replacement, and oversized-transient cases.
- Add an instrumentation-based regression test showing that a fitting insertion does not enumerate existing keys during eviction planning. The permitted final accounting reconciliation must be measured separately rather than mistaken for an eviction scan.
- Assert deterministic key order and exact `resident_bytes` after every complete operation, including failed insertion and replacement restoration.
- Add a scaling case that compares fitting bulk retention at `N` and `2N`; it must reject the current approximately fourfold growth pattern.

**Benchmark evidence**

Re-run retention with the actual 4,453 AAMP-shaped tiles and a budget large enough to avoid eviction. Record tile count, decoded payload bytes, retention duration, and whether eviction occurred. The benchmark-machine target is to reduce this phase from 846.93 milliseconds to tens of milliseconds, with approximately linear rather than quadratic growth when the tile count doubles. Also retain one smaller eviction-required benchmark so the fast no-eviction path does not hide a regression in protected LRU traversal.

**Implemented evidence (2026-08-29)**

`_evict_until_fits()` now returns from a constant-time capacity check before constructing an iterator or victim list. When eviction is necessary, it traverses entries lazily in LRU order, skips protected keys, stops after collecting enough unprotected payload bytes, and deletes the collected victims only after traversal. The final complete-operation byte reconciliation remains outside the insertion loop.

On the full-extent AAMP case, one cold `retain()` call kept all 4,453 tiles and 0.6925 MiB under the 1-GiB budget, proving that no eviction occurred. Retention fell from 846.93 milliseconds to 3.83 milliseconds, approximately a 221-fold improvement. The complete cold worker snapshot fell from 3,199.72 to 2,103.89 milliseconds; its remaining dominant cost is tile-major decoding rather than residency bookkeeping. The report is `/private/tmp/napari-harpy-slice3-cache-to-canvas-full-aamp.json`.

A fitting synthetic retention comparison measured 3.49 milliseconds for 4,453 one-point tiles and 6.76 milliseconds for 8,906 tiles, a 1.94-times increase when the tile count doubled. Deterministic unit instrumentation independently proves that the fit path performs no key or item traversal during eviction planning and visits each retained value exactly once in the final accounting reconciliation.

The eviction-required control used a centered 75-tile, 1,080-point AAMP viewport with a 1,024-byte residency budget. Only 19 tiles remained resident, so the control exercised eviction rather than the capacity fast path. Cold retention measured 0.10 milliseconds; the warm request protected its 19 hits, read the remaining misses, and completed retention in 0.30 milliseconds. The report is `/private/tmp/napari-harpy-slice3-cache-to-canvas-eviction-control.json`.

Focused validation passed with all 9 residency tests and all 28 cache-session tests. Coverage includes fitting bulk insertion, deterministic linear traversal counts, oldest-unprotected eviction after a protected prefix, insufficient protected capacity, successful replacement, failed-replacement restoration, exact byte accounting, oversized transient payloads, and end-to-end transient rendering.

**Exit condition**

CPU retention is no longer visible as a major cold-snapshot phase; fitting bulk insertion is near-linear; eviction order, protection and byte accounting remain exact; and warm CPU tile reuse is unchanged.

### Slice 4 — Enforce homogeneous bucket display batches

This slice makes the existing production invariant explicit before the selected-value physical-read path diverges. One bucket display batch has exactly one selection mode:

```text
complete batch -> every selected_value_ids is None
subset batch   -> every request has a nonempty selected_value_ids array
```

The normal viewport path already constructs homogeneous requests. `_ViewportReadPlan` records one plan-wide `requested_value_ids` mode and requires every tile's `applicable_value_ids` to agree with it. Filtering a plan to CPU-residency misses and grouping those requests by bucket preserve that mode. The singleton `read_display_payload()` wrapper also creates a one-request batch, which is homogeneous by definition.

The remaining ambiguity is local to `_BucketReader.read_display_payloads()`: its per-request optional selection type currently permits a caller to assemble a mixed complete/subset tuple even though no production plan does so. Supporting that hypothetical state would complicate the next slice with partial point-level ID reads and stitched output. This slice rejects it instead.

**Production changes**

1. At the beginning of `read_display_payloads()`, validate that `requests` is a nonempty tuple and that every member is a `(descriptor, selected_value_ids)` pair before interpreting the batch mode.
2. Require one homogeneous mode across the complete tuple: either every `selected_value_ids` is `None`, or none is `None`. Raise a clear `ValueError` for a mixed batch.
3. Complete this validation before allocating `batch_tile_indptr`, resolving complete or selected intervals, consulting lookup indexes, or accessing any point-payload Zarr array. Invalid input must have no planning, lookup, or physical-IO side effect.
4. Derive the complete-versus-subset mode once for the accepted batch. Later physical-read code may branch on that batch-level invariant and must not add mixed-mode stitching.
5. Update the method contract to state that per-tile selected arrays may differ because values occur in different tiles, but their presence or absence may not differ inside one batch.
6. Keep `_ViewportReadPlan`'s existing plan-wide validation as the upstream construction invariant. Do not duplicate the new bucket-boundary check in `_read_manifest_requests()` or add a new request dataclass merely to encode the same fact.

**Focused tests**

- Reject complete-then-subset and subset-then-complete batches with the same explicit validation error.
- Patch interval resolution, lookup access, and point-array access to fail if called, proving that mixed input is rejected before planning or physical IO.
- Retain successful multi-tile complete and proper-subset batch tests.
- Retain singleton complete and selected reads through the plural path.
- Retain `_ViewportReadPlan` tests proving that all-values plans contain only `None` selections and proper-subset plans contain only tile-applicable arrays.

**Benchmark evidence**

None is required. This is a fail-fast internal contract slice and does not change valid-request performance or physical payloads.

**Exit condition**

Every accepted bucket display batch has one explicit selection mode. Mixed complete/subset input fails before allocation, interval resolution, lookup access, or physical IO, so later read paths never need mixed-mode behavior.

### Slice 5 — Eliminate point-level `value_id` reads for proper subsets

**Status: Implemented**

This slice depends on the homogeneous-batch contract established by Slice 4 and reduces selected-value tile-major IO before introducing a new physical layout. A **proper subset** means one or more canonical values, but fewer than the complete value vocabulary. Selecting the complete vocabulary continues to normalize to the all-values path.

The cache contains two distinct kinds of value-ID data:

- `ranges/value_id` is compact lookup metadata with one value ID per nonempty tile/value run. It is loaded into `_BucketLookupIndex` together with `ranges/row_start` and `ranges/row_count` and remains the trusted source for resolving selected ranges.
- the point-level `value_id` array contains one value ID for every stored coordinate row. It is aligned with `location`, remains chunked on disk, and is the expensive payload this slice avoids reading for proper subsets.

This slice does not remove or bypass `ranges/value_id`. It uses that already validated, resident metadata to avoid decoding point-level IDs that the reader already knows.

**Pre-implementation and implemented read flow**

Before this slice, `read_display_payloads()` resolved every selected tile/value run to an exact bucket-global row interval, combined those intervals into one row selector, and applied the same selector to both `location` and the point-level `value_id` array. It then split the two aligned arrays back into per-tile `_PointDisplayPayload` values.

For example, if resident metadata resolves a request to:

```text
value 0 -> rows [0:2]
value 2 -> rows [3:5]
```

the pre-implementation path read location rows `[0, 1, 3, 4]` and also read their four point-level IDs from Zarr. The implemented path reads only those location rows and constructs the aligned IDs `[0, 0, 2, 2]` in memory from the two range values and counts. The returned payload is identical; only the physical source of its `value_id` buffer changed.

**Production changes**

1. Extend the internal selected-range result in `storage/bucket_reader.py` so every resolved interval retains the canonical value ID and row count that produced it, rather than returning only unlabelled `(start, stop)` bounds and a total count.
2. Preserve those labelled logical ranges independently of the physical row selector. `_exact_row_selection()` may merge touching intervals into one slice, but ID synthesis must still know where each selected value's output span begins and ends.
3. For proper-subset requests, apply the exact selector only to `location`. Allocate one aligned C-contiguous `uint32` output array and fill each span with its range's canonical value ID. Do not access the point-level `value_id` Zarr array on this path.
4. Preserve deterministic selected-value order inside each tile, bucket-local tile order inside each physical batch, and restoration of the original manifest request order across bucket batching. Missing selected values contribute no interval or output rows; a tile with no requested value continues to return `None` at the bucket-reader boundary.
5. Continue reading point-level `value_id` for all-values or complete-tile display requests. Those rows contain multiple values and the tile-major point array remains their canonical source.
6. Keep the external `_PointDisplayPayload`, `_TileReadResult`, CPU residency, snapshot, and renderer contracts unchanged. They still receive aligned `location` and `value_id` arrays with the same dtype, shape, order, immutability, and ownership behavior. CPU-residency and packed-render byte accounting therefore remain unchanged.

No cache schema change or cache rebuild is required. Existing tile-major-only caches already contain all range metadata needed for reconstruction.

**Focused tests**

- Update `test_bucket_reader.py` to patch the point-level `value_id` array so any access fails during proper-subset reads.
- Cover one value, several nonadjacent values, adjacent ranges, missing values, multiple tiles in one bucket, and restoration of request order.
- Compare reconstructed IDs and coordinates byte-for-byte with the existing canonical result on fixtures.
- Assert that all-values reads still access and return point-level IDs.

**Implemented validation (2026-08-31)**

The final focused run passed all 87 tests across `test_bucket_reader.py`, `test_reader.py`, and `runtime/test_cache_session.py`. The bucket-reader coverage makes point-level `value_id` access a hard failure for proper subsets; covers one value, nonadjacent values, adjacent ranges, missing values, and multiple tiles; compares reconstructed coordinates and IDs byte-for-byte with the canonical result; and proves that complete reads still access both `location` and point-level `value_id`.

**Benchmark evidence**

The required full-extent AAMP real-canvas run passed. It rendered the same 60,512 points across 4,453 logical tiles, reported `all_value_ids_match_selection=true`, and reduced point-level `value_id` Zarr calls from 69 to zero:

| Metric | Slice 3 baseline | Slice 5 |
|---|---:|---:|
| Point-level `value_id` calls | 69 | 0 |
| Point-level `value_id` time | 905.90 ms | 0 ms |
| `location` calls | 69 | 69 |
| `location` time | 1,047.53 ms | 1,102.38 ms |
| Cold worker snapshot | 2,103.89 ms | 1,259.25 ms |
| GUI activation | 0.547 ms | 0.498 ms |
| Warm draw median | 4.052 ms | 3.924 ms |

These cold measurements mean the first request in each process after lookup-index loading; they do not flush operating-system filesystem caches. The structural acceptance evidence is therefore the zero point-level calls together with the correct 60,512 returned IDs. The timing evidence is consistent with the removed read and shows no material GUI activation or rendering regression. The reports are `/private/tmp/napari-harpy-slice3-cache-to-canvas-full-aamp.json` and `/private/tmp/napari-harpy-slice5-cache-to-canvas-full-aamp.json`.

This is an IO optimization, not removal of value IDs from memory. The worker still constructs the same `uint32` IDs for CPU residency and render-batch packing. The 69 `location` calls and their tile-major chunk/shard amplification also remain. Slices 6 and 8 address that separate location-locality problem with the value-major sidecar and physical-payload routing.

**Exit condition**

Satisfied: proper-subset tile-major fallback performs coordinate-only physical reads and synthesizes IDs from already validated in-memory range metadata.

### Slice 6 — All-level value-major sidecar schema and writer

**Status: Implemented**

This slice makes the new physical ordering constructible, atomically published, and independently validated at the same structural/index boundary used by normal tile-major publication. It does not route viewer reads to it yet or perform coordinate-payload equivalence validation.

**Implemented storage contract**

The implemented cache has one strict `harpy-multiscale-points-zarr-cache-0.2` root contract. `_CacheAttributes.to_dict()` always emits that version, `_parse_cache_attributes()` requires its exact root-key set, and `_CacheRootReader` requires the five root groups `tile_major`, `values`, `manifest`, `value_tiles`, and `value_major`. At every level, rows in the original point payload are physically ordered by tile and then by `(value_id, point_id)` within each tile. The `value_tiles` catalog transposes compact range records into `(level, value_id, manifest_index)` order, but contains counts and tile references rather than coordinates.

Slice 6 adds a second location payload for every serialized level alongside the tile-major buckets. It does not replace `_BucketWriter`, alter the tile-major payload, or change the logical tile contract. Each level sidecar keeps the same tile-relative `(N, 2) float32` location representation and changes only physical row order:

```text
tile-major level L payload       tile_y -> tile_x -> value_id -> point_id
value-major level L sidecar      value_id -> manifest_index -> point_id
```

For example, if the Exact value-to-tile catalog contains:

```text
value 0 -> manifest 2 -> 2 points
value 0 -> manifest 8 -> 1 point
value 1 -> manifest 1 -> 2 points
```

the sidecar stores the first two locations for value 0/manifest 2, the next location for value 0/manifest 8, and then two locations for value 1/manifest 1. `value_point_indptr=[0, 3, 5]` addresses the complete per-value intervals. The existing `value_tiles` records and counts split each value interval back into its manifest tiles, so the sidecar does not need another `manifest_index` array.

Two distinct pointer tables participate in that reconstruction:

- the existing `value_tiles/indptr[level, value_id:value_id + 2]` selects the value's range-level catalog records in the aligned `value_tiles/manifest_index` and `value_tiles/n_points` arrays; and
- that level's `value_point_indptr[value_id:value_id + 2]` selects the value's point-level location interval in `value_major/level_L/location`.

For the example above, the aligned Exact catalog is conceptually:

```text
value_tiles/indptr[0]         = [0, 2, 3]
value_tiles/manifest_index    = [2, 8, 1]
value_tiles/n_points          = [2, 1, 2]
value_point_indptr            = [0, 3, 5]
```

The reader already knows the requested canonical `value_id`. For value 0, `value_point_indptr[0:2]` gives the complete location interval `[0, 3)`. Repeating that known ID three times constructs the point-aligned IDs without a sidecar `value_id` array. Starting at point row 0, the aligned catalog counts `[2, 1]` divide the interval into `[0, 2)` for manifest row 2 and `[2, 3)` for manifest row 8. For value 1, the point interval `[3, 5)` and count `[2]` identify manifest row 1. In pseudocode:

```python
point_cursor = value_point_indptr[value_id]

for manifest_index, n_points in value_tile_records(value_id):
    point_stop = point_cursor + n_points
    if manifest_index is required_for_viewport:
        location = sidecar_location[point_cursor:point_stop]
        value_ids = np.full(n_points, value_id, dtype=np.uint32)
        emit_logical_tile(manifest_index, location, value_ids)
    point_cursor = point_stop
```

`manifest_index` is therefore not eliminated from the cache. It remains stored once per value/tile range in the existing catalog and addresses the existing manifest descriptor, including the tile-grid coordinates needed to interpret tile-relative locations. It is merely not duplicated once per point in the sidecar. The sidecar writer must guarantee that location blocks follow exactly this catalog record order and that every block length equals its catalog `n_points`. Focused writer tests prove that ordering on small fixtures; the optional exhaustive validator in Slice 7 can prove location-for-location equivalence on a retained cache. Normal publication validation reconciles the structural and count contract without decoding locations.

Bridge and every Spatial level apply the same reconstruction against their own `value_tiles/indptr[level]`, manifest-record interval, `n_points` counts, and `value_major/level_L/value_point_indptr`; only the level point count and tile geometry differ.

The implemented slice ends at the storage boundary. `_PointsCacheReader`, viewport planning, CPU residency, render-batch packing, composition, and VisPy continue to use the tile-major path after Slice 6. Slice 8 introduces the post-LOD route decision and consumes the sidecar. Optional Slice 7 adds developer-only exhaustive equivalence validation without changing publication or runtime behavior.

**Schema decisions**

1. Store an explicit compact `value_major` descriptor in root cache metadata rather than inferring capability from directory presence. It records the group name, point row order, and the chunk/shard row settings shared by value-major point arrays. The root cache schema, serialized level metadata, and canonical sidecar schema define the required level set, array names, dtypes, and dimensions; strict hierarchy and layout validation enforce that contract without duplicating it in the descriptor.
2. Bump the cache schema outright and implement only the new contract. Do not add a compatibility parser for the preceding tile-major-only schema: pre-change caches are rejected and must be rebuilt. An unknown cache schema is likewise rejected.
3. Store the initial sidecar under one unambiguous generation-owned path such as:

   ```text
   value_major/
       level_0/
           location
           value_point_indptr
       level_1/
           location
           value_point_indptr
       ...
       level_N/
           location
           value_point_indptr
   ```

4. At every level persist only tile-relative `location` and compact `value_point_indptr`. Each pointer has shape `(value_count + 1,)` and dtype `uint64`; each `location` has shape `(level_point_count, 2)` and dtype `float32`. Do not duplicate point-level `value_id`, `point_id`, the manifest, or the value-to-tile catalog. A row's value is implicit in its pointer interval, manifest identity comes from the existing ordered `value_tiles` records, and `point_id` is used only to establish deterministic construction order.
5. Build sidecars for every serialized level unconditionally. There is no builder enable flag, per-level opt-out, disabled state, or tile-major-only form of the new schema. Sidecar location chunk/shard settings and the construction batch bound can remain explicit configuration with defaults rather than being inherited accidentally from tile-major buckets.

**Writer changes**

1. Add a dedicated storage writer rather than extending `_BucketWriter` with a second unrelated row order.
2. Build all level sidecars within `_write_staged_cache_catalog()`, after the catalog has finalized its ordered records and root metadata but before that function returns. `_build_points_cache_zarr()` then calls `_validate_staged_cache()`. The catalog therefore supplies the authoritative per-level value-to-tile ordering and counts before the sidecars are written, while neither artifact is public yet.
3. Reuse the existing catalog transpose rather than resolving every source range again after the catalog has discarded its sort permutation. The compact bucket-range iterator already reads each validated `ranges/row_start` together with `value_id`, `manifest_index`, and `row_count`. While the catalog sorts those records into `(value_id, manifest_index)` order, apply the same permutation to `row_start` and write it to a generation-owned, construction-only index. Its rows align one-for-one with the persisted `value_tiles/manifest_index` and `value_tiles/n_points` rows. Do not duplicate `manifest_index` or `n_points` in this temporary index: the manifest resolves each record's source bucket, and the catalog already supplies its point count.
4. Keep that construction-only row-address index outside the published Zarr hierarchy, under a unique path owned by the current cache generation. It must be removed on success and on failure, and it must not survive into staged validation or publication. It is an out-of-core transpose aid, not part of the cache schema or a viewer-runtime index.
5. For each serialized level, consume the aligned catalog records and temporary source row starts in bounded point batches. Within one batch, group records by source bucket, read the corresponding canonical location ranges through a bounded set of bucket handles, and scatter those reads into one bounded location buffer in catalog order. Then append that buffer as one contiguous interval to the level's value-major `location` array. If one range exceeds the configured point bound, split that range without changing its point order. Because bucket point rows are already ordered by `(value_id, point_id)` within each tile, construction does not need to read or retain point IDs; `point_id` establishes the existing deterministic source order but is not persisted in the sidecar.
6. Construct each sidecar out of core. Bound the temporary row-address writes, location buffers, and retained bucket handles explicitly; do not materialize a complete level location array, point IDs, or all source rows in memory. Finish, reconcile, and release one level before advancing to the next. Cache-construction time is secondary to runtime locality but unbounded RAM and one lookup or Zarr operation per value/tile range are not acceptable.
7. Track buffered and physically written location rows explicitly; their sum is the accepted input-row count. At every level, reconcile each value pointer interval with that level's catalog count, reconcile the final pointer with that level's manifest total, and require the final writer count to equal the declared level point count before closing the sidecar. Normal staged validation independently verifies the physical `location` shape from Zarr metadata.
8. Include sidecar files in staging validation and atomic publication. Any sidecar write or validation failure must leave the preceding completed generation recoverable.
9. Thread explicit sidecar location chunk/shard settings, the construction point-batch bound, and the retained bucket-handle bound through the public builder configuration and `scripts/build_tiled_points_cache_variant.py` so the supplied-cache build is reproducible from one recorded command. Do not expose a global or per-level switch that omits a sidecar.

The mandatory staged validator must remain aligned with the current production tile-major validation policy rather than becoming a full payload verifier. It must reopen the staged generation without accepting writer results, make the hierarchy and root descriptor aware of exactly one `value_major/level_L` group for every serialized level, reject missing or extra levels, groups, arrays, attributes, or schema fields, and validate the declared dtype, shape, chunks, shards, codec, fill value, and chunk-key encoding of every sidecar array.

For each level, normal publication validation reads `value_point_indptr` completely and requires origin zero, nondecreasing pointers, per-value pointer differences equal to that level's aggregated `value_tiles/n_points`, and a terminal equal to both the level point count and the metadata-declared `location` row count. It validates the `location` array's physical contract from Zarr metadata but does not index or decode location rows and does not compare them with tile-major locations. Consequently, just like current production validation of tile-major `location`, it is not expected to detect a missing or undecodable location shard or semantically incorrect location values. Writer-time cursor reconciliation, focused location-order tests, and optional Slice 7 exhaustive validation cover those distinct concerns. This extends the existing build transaction rather than creating a second publication step:

```text
Exact -> Bridge/Spatial -> catalog -> mandatory sidecar for every level
      -> independent staged validation -> mark complete -> atomic publication
```

**Focused tests**

- Add small-fixture writer tests that compare every sidecar location block with its expected tile-major range and cover ordering, value pointers, empty values, several tiles per value, deterministic point order, changing level geometry, chunk boundaries, and per-level final row-count reconciliation.
- Add publication-validation corruption tests for missing and extra levels, metadata, dtype, shape, pointer monotonicity, pointer terminal value, catalog-count disagreement, and metadata-declared coordinate count at Exact, Bridge, and Spatial levels.
- Prove that normal staged validation validates `location` layout without decoding it, matching the existing tile-major publication contract.
- Extend builder and staging-validation tests to cover successful all-level publication, rollback on any level-sidecar failure, missing mandatory sidecar metadata or arrays, and rejection of the pre-change tile-major-only schema.
- Verify that an unrecognized sidecar schema is rejected rather than ignored.

**Construction evidence**

Build the supplied cache with every level sidecar and record total construction duration, peak RSS, logical bytes, actual compressed physical bytes, total cache size, and compression ratio. Construction duration is reported but is not a rejection criterion unless operationally prohibitive.

**Measured full-cache build — 2026-09-02**

The current implementation rebuilt the canonical cache from `points/transcripts_global_ROI1/points.parquet`: 136,578,750 source points, 5,122 canonical values, and nine serialized levels. The build used the current defaults: 512-unit leaf tiles, a 100,000-point overview budget, two Dask workers, 2,000,000 target points per tile-major bucket, Zstd, 4,096-row point chunks, 131,072-row point shards, a 1,048,576-point value-major construction-batch bound, and `max_open_value_major_readers=None`. The latter retains all source-bucket readers for the active level and releases them before advancing to the next level.

| Measurement | Result |
|---|---:|
| Source validation | 2.48 s |
| Complete builder, excluding source validation | 750.39 s (12 min 30 s) |
| Peak process RSS, sampled every 0.25 s | 4,050,288,640 bytes (3.77 GiB) |
| Incremental peak RSS above the pre-build baseline | 3,678,502,912 bytes (3.43 GiB) |
| Complete cache physical size | 2,893,702,906 bytes (2.70 GiB) |
| Complete cache file count | 8,520 |
| Value-major logical payload | 1,488,818,048 bytes (1.39 GiB) |
| Value-major compressed physical size | 1,203,063,040 bytes (1.12 GiB) |
| Value-major logical-to-physical ratio | 1.238:1 |

The resulting `harpy-multiscale-points-zarr-cache-0.2` generation `939bdb3e-d137-49d5-9d3e-0779c67e4156` was independently reopened as `complete`. Its root contains exactly `manifest`, `tile_major`, `value_major`, `value_tiles`, and `values`; publication left no staging generation or build lock behind.

**Exit condition**

Every completed cache generation using the current schema contains a value-major location sidecar for every serialized level, and normal publication validation independently verifies its mandatory hierarchy, layout, pointer, and catalog-count contract without decoding location payloads. Pre-change tile-major-only or partially covered caches are rejected and must be rebuilt.

### Slice 7 — Optional exhaustive value-major location-equivalence validation

**Status: Implemented**

This is a developer-only validation layer for format changes, release qualification, or investigation of suspected corruption. Its command-line target is an explicitly completed, retained cache generation. It is deliberately excluded from normal cache construction and publication, just as the existing exhaustive tile-major validator is separate from `_validate_staged_cache()`.

**Scope**

1. Extend `scripts/validate_multi_scale_cache_points_zarr_exhaustive.py` rather than adding coordinate decoding to the production staged validator. The CLI must require `publication_state="complete"`; it is a post-publication diagnostic for a retained cache, not a hook into the builder's private staging generation.
2. Extract the common read-only hierarchy, layout, catalog, manifest, bucket-range, and artifact checks behind a private helper that requires an explicit expected publication state. Keep `_validate_staged_cache()` as the unchanged production-facing wrapper that passes `staging`, and add a strict completed-generation wrapper for the developer tool that passes `complete`. Do not infer or silently accept either state. Run that completed-generation validation first, then retain the existing exhaustive tile-major payload, point-identity, cross-level, and optional source-equivalence checks.
3. For every serialized level, reuse the existing levelwise range-reconciliation pattern: stream each bucket's persisted sparse ranges once, retain only compact `value_id`, `manifest_index`, `row_start`, and `row_count` metadata for the current level, and sort that metadata into the persisted catalog's `(value_id, manifest_index)` order. Do not perform one independent sparse-index lookup or Zarr operation for every catalog record.
4. Consume those ordered source ranges in bounded point batches, read their canonical tile-major location intervals through a bounded bucket-reader cache, and compare them exactly with the corresponding `value_major/level_L/location` blocks. Use a validator-only default bound of 1,048,576 compared points per batch and expose the bound as a function argument so focused tests can force smaller cross-chunk batches; it is not cache metadata or an application setting. Equality of the complete ordered location sequence proves both membership and the sidecar's inherited `(value_id, manifest_index, point_id)` order even though point IDs are not duplicated in the sidecar. Process and release one level at a time; optional exhaustive validation may be IO-expensive, but it must not require a complete level's locations in RAM.
5. Decode every sidecar location row. Missing or corrupt location chunks, swapped value/manifest blocks, incorrect locations, truncated output, and ordering errors must fail this optional path.
6. Do not call this validator from `_build_points_cache_zarr()`, do not make publication depend on it, and do not add an application setting that enables it implicitly.

The exhaustive location-equivalence implementation remains in `scripts/validate_multi_scale_cache_points_zarr_exhaustive.py`. Add `tests/multi_scale_cache_points_zarr/test_exhaustive_validation.py` only as the dedicated test module for that developer script: it may call the script's focused internal validation functions directly and exercise the CLI where useful, but it must not become a second implementation or move exhaustive payload validation into the installed production package. The only installed-package change in this slice is the small shared read-only validation refactor needed to preserve strict `staging` and `complete` wrappers.

**Focused tests**

- Compare complete Exact, Bridge, and Spatial sidecars with their tile-major sources on a small multilevel fixture.
- Corrupt one location, swap equal-sized catalog blocks, remove a sidecar location shard, and disturb a block boundary; prove that normal publication validation retains its metadata-only payload policy while the exhaustive validator rejects each corruption.
- Exercise an empty value interval and batching across sidecar chunk boundaries.
- Prove that the production staged wrapper still requires `staging`, the developer wrapper requires `complete`, and neither automatically accepts the other publication state.
- Prove that the exhaustive path remains bounded and leaves its caller-owned temporary root intact and empty after success or failure.

**Optional evidence**

Run the exhaustive comparison once on a retained all-level build of the supplied cache and report duration, peak RSS, logical coordinate bytes compared, and compressed bytes covered by the compared location arrays. Do not label operating-system or Zarr-cache effects as measured physical reads without store-level instrumentation. These measurements characterize the developer tool; they are not publication or runtime acceptance gates.

The implemented CLI completed successfully against generation `939bdb3e-d137-49d5-9d3e-0779c67e4156`, covering all nine levels and 186,056,149 location rows in each physical ordering. The measurement used the default 1,048,576-point comparison-batch bound and did not include source-Parquet equivalence.

| Measurement | Result |
|---|---:|
| Complete exhaustive CLI duration | 642.44 s (10 min 42 s) |
| Peak child-process RSS | 4,290,740,224 bytes (4.00 GiB) |
| Logical location payload per ordering | 1,488,449,192 bytes (1.39 GiB) |
| Combined tile-major and value-major logical location bytes compared | 2,976,898,384 bytes (2.77 GiB) |
| Compressed value-major location-shard bytes covered | 1,202,957,621 bytes (1.12 GiB) |

The compressed figure is the on-disk size of the sidecar location shard files covered by the comparison, not an instrumented physical-read count. The CLI also performs its pre-existing structural, tile-major payload, point-identity, and cross-level proofs, so 642.44 seconds must not be interpreted as an isolated value-major comparison time.

**Exit condition**

An explicitly invoked developer tool can require a completed retained cache and prove location-for-location equivalence between every value-major sidecar and its canonical tile-major payload without changing the normal builder, the strict staging-state publication gate, or viewer runtime.

### Slice 8 — Post-LOD physical-payload routing and sidecar reads

This slice realizes the cold-read improvement while preserving one logical tile/snapshot contract above the reader.

**Current-to-target interpretation**

This is a physical reader change, not an LOD, CPU-residency, snapshot, or renderer change. Today `_read_viewport_snapshot()` first calls `_PointsCacheReader.select_level()`, returns a metadata-only empty snapshot immediately when the selected level is over budget, constructs `_ViewportReadPlan` only for an accepted level, reuses CPU-resident logical tiles, and passes only the missing logical tile keys to `read_planned_tiles()`. That order remains unchanged.

The current `read_planned_tiles()` has one physical implementation for every accepted selection. It restores the plan's manifest rows, groups them by `(level, bucket_id)`, and calls `_BucketReader.read_display_payloads()` once per bucket. For a proper subset, each bucket reader resolves the requested tile/value pairs through its resident `ranges/{tile_indptr,value_id,row_start,row_count}` metadata and applies the resulting sparse row selector to that bucket's tile-major `location` array. Slice 5 already synthesizes the aligned output IDs instead of reading point-level `value_id`, but the coordinate read still touches tile-major chunks distributed across the many positive tiles.

Slice 8 changes that accepted-read portion to:

```text
selected values and viewport
            |
            v
       select_level()                       unchanged semantic LOD choice
            |
      over budget? ---- yes ----> metadata-only snapshot; no read plan or payload
            |
            no
            v
       plan_viewport()                      positive logical tiles plus one route
            |
            v
       CPU-residency lookup                 unchanged; read only missing tiles
            |
       +----+-------------------------+
       |                              |
       v                              v
all values                      proper subset
tile-major buckets              selected level's value-major sidecar
       |                              |
       +---------------+--------------+
                       v
              ordered _TileReadResult values
                       |
                       v
          existing residency, render batch, and VisPy path
```

There are conceptually three outcomes—no payload, tile-major, and value-major—but the current worker rejects the over-budget case before constructing `_ViewportReadPlan`. Preserve that useful early return. The plan itself therefore needs only two explicit physical routes, for example `tile_major_all_values` and `value_major_subset`; it must not invent a nominal `no_payload` plan that can never be read. The complete vocabulary already normalizes to the all-values `None` selection, so a non-`None` `_SelectedValueIndex` unambiguously denotes the proper-subset route.

**Production changes**

1. Keep `select_level()` unchanged and execute it before physical-payload selection.
2. Preserve the existing over-budget early return before `plan_viewport()`. Extend the generation-bound `_ViewportReadPlan` for accepted reads with exactly one explicit physical route:

   ```text
   requested_value_ids is None         -> tile_major_all_values
   proper selected-value index         -> value_major_subset
   ```

3. Make the route decision once per plan in `_PointsCacheReader`; do not decide independently in the GUI, cache session, individual bucket readers, or VisPy. The route is chosen only after LOD selection, so Exact, Bridge, and every Spatial level use the sidecar belonging to the level that was actually selected.
4. Keep `read_planned_tiles(plan, tile_keys_to_read)` as the single physical-read dispatch boundary. CPU residency is evaluated before this call, so the value-major path must read only requested nonresident tiles rather than rereading every positive tile in the viewport. For a proper subset, the plan must retain a reference to the immutable `_SelectedValueLevelIndex` used to construct it; do not copy its arrays, add mutable selected-value state to `_PointsCacheReader`, or force `read_planned_tiles()` to reload catalog records. An all-values plan retains no selected-level index. Validate this private plan field against the plan's generation, level, requested IDs, and route.
5. Add a dedicated value-major sidecar reader under the storage layer. Reuse the strict sidecar arrays already opened by `_CacheRootReader`; do not reopen a store per tile. During `_PointsCacheReader` entry, materialize every level's compact `value_point_indptr` vector once and retain it for that reader's lifetime, including its bytes in `resident_index_bytes`. For the supplied nine-level, 5,122-value cache this is only 368,856 bytes. Keep `location` as an on-disk Zarr array. A full-extent one-value read should reduce to one basic sidecar interval whenever all of that value's records are requested.
6. Use the existing `_SelectedValueLevelIndex` arrays. Its `value_indptr` partitions the selected values, while aligned `manifest_index` and `n_points` identify every tile record and its point count. These arrays already retain all records for each selected value at the chosen level, including records outside the current viewport, so no bucket sparse-range lookup is needed to derive sidecar addresses.
7. Derive per-record sidecar offsets from the value's base pointer and an exclusive cumulative sum of its complete ordered `n_points` records. Do not introduce a persisted or cache-wide resident `record_point_indptr`. For example:

   ```text
   value A sidecar base = 1,000

   ordered value_tiles records:
       manifest tile 10: 3 points  -> sidecar [1000:1003]
       manifest tile 20: 5 points  -> sidecar [1003:1008]
       manifest tile 30: 2 points  -> sidecar [1008:1010]

   viewport requests only tile 20  -> read [1003:1008], not [1000:1005]
   ```

   A partial viewport or CPU-residency subset may discard tile 10 from physical output, but its three rows must still advance the cursor. Computing the prefix from visible or missing records alone would address the wrong sidecar rows.
8. Combine adjacent selected sidecar intervals into basic slices where possible and otherwise use bounded exact row selections against the one level-wide `location` array. Associate each returned run with its known `(value_id, manifest_index)`, then scatter those runs into per-tile output buffers. A tile containing several selected values receives its value blocks in increasing value-ID order; point order inside each value/tile block is already preserved by the sidecar. This reconstructs the same selected tile order as the tile-major `(value_id, point_id)` payload.
9. Construct aligned `uint32 value_id` rows from the known value blocks; never read a point-level value-ID array from either physical ordering. Restore the original plan's manifest/spatial tile order and return exactly the existing `_TileReadResult(level, tile_x, tile_y, tile_size, location, value_id)` contract, including correct empty intersections and cache-origin semantics.
10. Preserve cooperative cancellation and generation checks across multi-run sidecar reads. Thread the worker's raising cancellation callback into `read_planned_tiles()` and check it between bounded selections; diagnostic callers may omit it. A Zarr operation already in progress remains non-interruptible, and no result from an obsolete generation or selection may be published.
11. Keep `_read_viewport_snapshot()`, `TileResidencyKey`, `_CpuTileResidency`, `TiledPointsRenderTile`, worker-side render-batch packing, snapshot delivery, composition, and VisPy unaware of which physical payload supplied a tile.
12. Expose route, sidecar selection count, touched chunks/shards, selected rows, decoded rows, and physical bytes in benchmark diagnostics.

**Slice boundary with Slices 9 and 10**

Slice 8 makes proper-subset viewport payload reads independent of bucket sparse ranges, but its first implementation retains the visible value-to-tile relation twice: the immutable selected-value level index remains in the plan for sidecar addressing, while each planned tile also carries an `applicable_value_ids` array. Slice 9 removes that duplicate projection without changing the physical route. The current startup sequence still projects and eagerly loads every bucket lookup index; its approximately 8.1-second startup cost and 568.4-MiB resident lookup allocation therefore remain temporarily even though the selected-value payload path no longer consumes them. Slice 10 removes that startup policy, separates compact complete-tile addressing from sparse range metadata, and then removes `max_bucket_lookup_bytes` from the viewer settings. Do not silently broaden Slice 8 into either follow-up cleanup.

**Focused tests**

- Prove that physical routing does not alter the LOD decision and that every proper-subset Exact, Bridge, and Spatial plan selects its corresponding sidecar.
- Cover proper-subset sidecar routing and all-values tile-major routing at Exact, Bridge, and representative Spatial levels.
- Compare sidecar and the pre-change tile-major results for one value, several values, full extent, partial viewport, CPU-residency misses, and empty intersections at multiple levels.
- Assert identical tile keys, tile order, coordinates, value IDs, estimated counts, omitted values, and cache-origin behavior.
- After normal Slice 8 startup, patch bucket selected-range resolution and tile-major point-array access to fail and prove that a proper-subset viewport read does not touch them. Do not claim that sparse indexes are absent from memory until Slice 10 removes the eager startup load.
- Exercise cancellation and stale generations during a multi-run sidecar read.

**Benchmark evidence**

For full-extent AAMP at Exact:

- the route is `value_major_subset`;
- selected coordinate rows remain 60,512;
- touched `location` chunks fall from 4,291 toward the projected 16 rather than remaining proportional to positive tiles;
- no point-level `value_id` array is decoded;
- no bucket sparse-range array is consulted by the request, although the current eager startup policy still leaves those indexes resident until Slice 10; and
- cold selected-value payload time improves materially from the 4.14-second aligned-array baseline. A practical prototype target is below one second on the same benchmark machine and filesystem state, but the report must retain raw chunk, byte, and call evidence rather than accepting wall time alone.

The pre-change full-extent ADAMTS1 benchmark establishes that coarser-level amplification is real rather than hypothetical. It selects Bridge with 92,499 points across 5,853 tiles, touches 5,133 coordinate chunks, decodes approximately 21.0 million coordinate rows for 92,499 returned rows (227-times amplification), spends 944 ms in `location`, and takes 1.14 seconds for the cold worker snapshot even after bucket sparse-range indexes are resident. The report is `/private/tmp/napari-harpy-bridge-adamts1-assessment.json`.

Repeat that case through the Bridge sidecar and add proper-subset requests that deliberately select representative Spatial levels. At every selected level, report selected and decoded rows, chunks, shards, physical bytes, and wall time, and prove that the sidecar request does not consult bucket sparse ranges. Report the still-existing eager startup load separately rather than attributing it to the payload request. Also benchmark multiple genes, a partial viewport, and all values to ensure the all-level sidecars do not regress the tile-major all-values path.

**Exit condition**

The reader chooses physical locality after semantic LOD selection and returns the existing logical tile contract. Every proper-subset level uses its mandatory sidecar, all-values requests retain tile-major routing, and no proper-subset viewport payload read resolves or reads bucket sparse ranges. Removing duplicate plan membership is explicitly deferred to Slice 9; removing eager viewer-startup loading and resident sparse-index footprint is explicitly deferred to Slice 10.

### Slice 9 — Remove duplicated per-tile selected-value membership

**Status: Implemented**

This is a bounded internal reader refactor following the first value-major runtime implementation. It does not change cache contents, semantic LOD selection, physical routing, CPU residency, logical tile output, render-batch packing, or VisPy.

Before this slice, the proper-subset plan retained two orientations of the same visible membership relation:

```text
_SelectedValueLevelIndex
    selected value -> every manifest row and point count at the level
        |
        | plan_viewport(): intersect with visible manifest rows
        v
_PlannedTileRead.applicable_value_ids
    positive visible manifest row -> selected values in that tile
        |
        | CPU residency chooses missing planned tiles
        v
_read_value_major_requests(): transpose again
    selected value -> missing manifest rows
```

The selected level index is not copied: `_ViewportReadPlan` retains the same immutable object by reference. Its complete level-wide `n_points` sequences must remain available because sidecar offsets include records preceding the visible or nonresident records. The removed duplication was the per-tile `applicable_value_ids` projection. It created a dictionary and potentially thousands of small NumPy arrays for every proper-subset viewport plan, including warm requests for which CPU residency later eliminates every physical read.

**Production changes**

1. Remove `applicable_value_ids` from `_PlannedTileRead` together with its validation, documentation, and plan-consistency branches. A planned tile retains only the generation-bound plan's logical level and coordinates plus its manifest and bucket identity.
2. Keep `requested_value_ids` and the selected level's exact immutable `_SelectedValueLevelIndex` reference in `_ViewportReadPlan`. They serve different contracts: the tuple is the canonical logical selection and positional key for `value_indptr`, while the index is the complete physical value-to-manifest relation and aligned point counts. Do not copy its arrays, retain the complete all-level `_SelectedValueIndex` in each plan, or move mutable selection state into `_PointsCacheReader`.
3. For a proper subset, make `plan_viewport()` derive only the sorted union of positive visible manifest rows from the resident selected-value index. Construct one `_PlannedTileRead` per positive logical tile without materializing a manifest-row-to-values dictionary or one NumPy value-ID array per tile. Planning must continue to perform no catalog or point-payload IO.
4. Preserve CPU-residency lookup before physical addressing. Pass only missing logical tile keys to `read_planned_tiles()` as today; do not precompute complete sidecar block lists for visible tiles that may already be resident. At the physical-reader dispatch boundary, reduce the filtered `_PlannedTileRead` objects to their sorted missing `manifest_row` tuple.
5. Do not pass both the complete `_ViewportReadPlan` and a filtered request subset to `_read_value_major_requests()`. The helper does not consume `plan.requests`, and that signature obscures the distinction between every positive viewport tile and only the CPU-residency misses. Give the helper the explicit inputs it consumes:

   ```python
   def _read_value_major_requests(
       self,
       *,
       level: int,
       requested_value_ids: tuple[int, ...],
       selected_value_level_index: _SelectedValueLevelIndex,
       manifest_rows: tuple[int, ...],
       raise_if_cancelled: Callable[[], None] | None,
   ) -> _ViewportReadResult:
   ```

   Keep `requested_value_ids` explicit: `_SelectedValueLevelIndex.value_indptr` is partitioned by selected-value position and deliberately does not duplicate the corresponding canonical IDs. `read_planned_tiles()` remains the plan-aware route dispatcher and passes these aligned fields from the already validated plan; the physical helper accepts neither `_ViewportReadPlan` nor `_PlannedTileRead`.
6. In `_read_value_major_requests()`, construct a sorted array from the requested missing `manifest_rows` and intersect it directly with each selected value's `manifest_index` interval. Use the aligned complete `n_points` interval and `value_point_indptr` base to derive exact sidecar blocks in value-major order. Values with no missing match produce no block; every requested positive tile must receive at least one block.
7. Preserve the existing scatter into manifest tile order and canonical increasing value-ID order within each tile. The resulting `_TileReadResult` must be byte-equivalent to the Slice 8 implementation.
8. Keep the all-values physical behavior unchanged, but rename `_read_manifest_requests()` directly to `_read_complete_tile_major_requests()` with no compatibility alias. Give it complete-tile inputs such as `(level, manifest_rows)` rather than `(manifest_row, selected_value_ids)` pairs. The word `complete` distinguishes this physical reader from both the historical sparse selected-range path and Slice 11's later in-memory filtering step. If the lower-level bucket API still represents a complete-tile request with `selected_value_ids=None`, construct that sentinel only at the bucket call boundary; do not retain it in `_PlannedTileRead`. Slice 11's `tile_major_filter` route will use this same complete-tile reader before filtering, so its interface must not imply that a future proper-subset route needs per-tile applicable-value arrays.
9. Remove the current cancellation asymmetry while changing the complete-tile manifest-reader signature. Forward `raise_if_cancelled` into that path and check it before and after every sequential bucket batch, matching the value-major reader's cooperative boundary. An individual Zarr operation remains non-interruptible, but cancellation must prevent later buckets from being read and must be observed before a tile-major result can return.
10. Remove or narrow helpers and benchmark code whose only purpose was to materialize the discarded manifest-row-to-values mapping. Keep the summary-only LOD path resident and free of catalog IO.
11. Do not add `tile_major_filter`, a route estimator, coverage expansion, debounce, or a new cache-format field in this slice. Those remain separately reviewable downstream work.

**Focused tests**

- Proper-subset plans contain exactly the positive visible logical tiles in manifest spatial order and retain no tile-specific value-ID arrays.
- The plan retains the exact selected level-index object rather than copying its NumPy arrays, and all-values plans retain no selected level index.
- One-value and multi-value reads over complete and partial viewports return the same ordered tile keys, `location`, and aligned `uint32 value_id` payloads as the pre-refactor value-major path at Exact, Bridge, and representative Spatial levels.
- CPU-resident tiles are removed before value-major block resolution; a fully resident request performs no sidecar addressing or payload read.
- Value-major dispatch passes explicit `level`, `requested_value_ids`, the exact selected level-index object, and only the missing manifest rows. `_read_value_major_requests()` accepts neither the complete `_ViewportReadPlan` nor `_PlannedTileRead` objects.
- Off-screen and resident records preceding a requested record still contribute to its sidecar prefix, proving that direct missing-row intersection does not shorten the complete per-value count sequence.
- Empty intersections, values absent from a selected level, stale generations, invalid tile keys, cancellation, and physical read failures preserve existing behavior.
- All-values reads remain on the tile-major path through the narrowed complete-tile manifest reader. Where the existing lower-level bucket API is retained, it receives `None` for every complete-tile request without that sentinel being stored in the viewport plan.
- Tile-major cancellation is checked before and after each bucket batch. A callback that raises after one bucket prevents every later bucket read, returns no partial result, and preserves the existing worker publication boundary.
- Patch catalog-array reads and bucket sparse-range resolution to fail during planning and proper-subset payload reads, proving the refactor remains entirely resident-index and sidecar based.

**Benchmark evidence**

Compare before and after on cold, partially resident, and fully resident requests with the same selections, viewports, and selected LODs. Report viewport-plan wall time, physical block-resolution time, total worker time, positive and missing tile counts, selected value/tile record count, plan-owned NumPy allocation count and bytes, sidecar reads, and returned bytes. Include a highly fragmented request with thousands of positive tiles and a multi-value request.

Acceptance requires eliminating tile-count-proportional value-ID arrays from the plan, preserving byte-equivalent payloads, and avoiding a material regression in cold sidecar reads. A fully resident request should show reduced or unchanged planning time; do not claim a latency improvement without the allocation and timing measurements.

**Implementation evidence — 2026-09-07**

The implementation removes the per-tile projection, dispatches explicit selected-level facts and missing manifest rows, and narrows/renames the complete tile-major helper without an alias. Cancellation is checked before and after each tile-major bucket batch. The 104 focused reader, value-major viewport/location, and cache-session tests pass; changed Python files pass Ruff.

`scripts/benchmark_tiled_points_viewport_planning.py` captures the real worker path before and after the refactor, including block resolution before location IO and plan-owned NumPy arrays separately from the borrowed level index. Measurements below are medians of five requests per case/state, with a 100,000-point budget. “Cold” means empty CPU residency, not a flushed filesystem cache; partial residency retains alternating planned tiles. Startup/index loading is measured separately. The after run was repeated without concurrent tests. No physical draw was benchmarked.

| Case | LOD | Snapshot points | Positive tiles | Selected level value/tile records | Plan-owned arrays before → after | Array data bytes before → after |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| AAMP, full extent | Exact | 60,512 | 4,453 | 4,453 | 4,453 → 0 | 17,812 → 0 |
| AAMP, centered 0.2 viewport fraction | Exact | 4,846 | 247 | 4,453 | 247 → 0 | 988 → 0 |
| AAMP + ADAMTS1, full extent | Bridge | 99,893 | 6,002 | 9,692 | 6,002 → 0 | 38,768 → 0 |
| AAMP + ADAMTS1, centered 0.2 viewport fraction | Exact | 45,472 | 337 | 10,393 | 337 → 0 | 2,328 → 0 |

Array data bytes exclude NumPy/Python object overhead and transient allocations. Planned tile identity objects and the borrowed selected-level index remain; this does not make the entire plan allocation-free.

| Case | CPU residency | Missing tiles | Plan ms, before → after | Block resolution ms, before → after | Total worker ms, before → after |
| --- | --- | ---: | ---: | ---: | ---: |
| AAMP, full | Cold | 4,453 | 16.69 → 5.99 | 5.50 → 5.35 | 105.90 → 97.88 |
| AAMP, full | Partial | 2,226 | 17.33 → 6.00 | 2.77 → 2.61 | 83.37 → 77.78 |
| AAMP, full | Full | 0 | 16.95 → 6.17 | 0 → 0 | 51.05 → 43.58 |
| AAMP, 0.2 | Cold | 247 | 0.97 → 0.36 | 0.33 → 0.31 | 7.81 → 8.12 |
| AAMP, 0.2 | Partial | 123 | 0.96 → 0.39 | 0.18 → 0.19 | 6.63 → 6.69 |
| AAMP, 0.2 | Full | 0 | 0.97 → 0.36 | 0 → 0 | 2.97 → 2.52 |
| AAMP + ADAMTS1, full | Cold | 6,002 | 25.70 → 8.49 | 13.00 → 12.22 | 173.54 → 149.63 |
| AAMP + ADAMTS1, full | Partial | 3,001 | 25.90 → 8.14 | 6.36 → 5.95 | 131.08 → 114.39 |
| AAMP + ADAMTS1, full | Full | 0 | 25.51 → 8.61 | 0 → 0 | 75.24 → 58.45 |
| AAMP + ADAMTS1, 0.2 | Cold | 337 | 1.59 → 0.54 | 0.77 → 0.78 | 19.79 → 18.25 |
| AAMP + ADAMTS1, 0.2 | Partial | 168 | 1.49 → 0.49 | 0.44 → 0.38 | 16.57 → 15.52 |
| AAMP + ADAMTS1, 0.2 | Full | 0 | 1.48 → 0.50 | 0 → 0 | 4.65 → 3.80 |

All 60 paired requests preserve the exact packed-vertex SHA-256, LOD/omission metadata, positive/missing tile counts, physical selection counts, returned bytes, and touched chunks/shards. Each cold or partial request performs one location selection; every fully resident request performs zero block-resolution calls and zero payload reads. Returned location payloads are unchanged:

| Case | Cold rows / bytes | Partial rows / bytes | Chunks / shards, both states |
| --- | ---: | ---: | ---: |
| AAMP, full | 60,512 / 484,096 | 30,325 / 242,600 | 16 / 1 |
| AAMP, 0.2 | 4,846 / 38,768 | 2,393 / 19,144 | 7 / 1 |
| AAMP + ADAMTS1, full | 99,893 / 799,144 | 50,373 / 402,984 | 25 / 3 |
| AAMP + ADAMTS1, 0.2 | 45,472 / 363,776 | 22,462 / 179,696 | 32 / 4 |

Planning improves in every measured case. Cold sidecar IO is structurally unchanged; location-reader medians vary by approximately -0.5 to +0.8 ms across these cases. The small AAMP viewport's cold worker median increases by 0.31 ms, so do not claim every measured latency improved. The main result is elimination of per-tile value arrays and a substantial planning reduction for fragmented requests, without an observed material cold-read regression.

Raw reports: `/private/tmp/napari-harpy-viewport-planning-before.json` and `/private/tmp/napari-harpy-viewport-planning-after-isolated.json`. These include individual measurements, startup costs, returned bytes, selection/chunk statistics, and payload hashes.

**Exit condition**

The immutable selected level index is the single authoritative selected-value-to-manifest relation. `_ViewportReadPlan` retains only the logical selection, that shared level-index reference, and ordered tile identities; the plan-aware dispatcher passes explicit selected-level facts and missing manifest rows into a value-major physical helper that accepts no plan or planned-tile object. Value-major sidecar blocks are derived directly for CPU-residency misses without a per-tile value-membership projection or reverse transposition. Both physical branches observe cooperative cancellation between their bounded sequential Zarr operations.

### Slice 10 — Remove bucket sparse-range indexes from the viewer runtime

**Status: Implemented**

This is primarily a startup-time and resident-memory improvement, not a renderer change. It removes the eager bucket lookup policy associated with the earlier approximately 8.1-second startup and 568.4-MiB lookup allocation, without replacing it with a fallback-index cache. Those are baseline measurements, not a promise that the entire previous startup duration disappears. All-level sidecars make the large sparse ranges a build-time and validation structure rather than a viewer-runtime resource.

**Starting point before implementation**

`_TiledPointsCacheWorker.start()` called `project_bucket_lookup_index_bytes()` and then `load_bucket_lookup_indexes()` across every level before announcing readiness. Projection itself opened the bucket readers to inspect their metadata; loading then retained all five arrays bundled in `_BucketLookupIndex`: `tile_offset` and `ranges/{tile_indptr,value_id,row_start,row_count}`.

Proper-subset viewport reads already used `_read_value_major_requests()` without consuming these bucket ranges. However, all-values reads went through `_read_complete_tile_major_requests()` → `_BucketReader.read_display_payloads()` → `resolve_complete_tile_interval()`. That final method required `_BucketLookupIndex` merely to access its `tile_offset` array. Deleting startup loading alone would therefore have broken all-values reads.

The central refactor is to separate **where a complete tile begins and ends** from **where particular values occur inside that tile**. The viewer still needs the first form of addressing, but no longer needs the second.

For an all-values tile occupying bucket point rows `[100:110)`, the reader needs only the equivalent of `location[100:110]` and point-level `value_id[100:110]`. It does not need to discover the individual value ranges inside that interval. `tile_offset` therefore remains necessary as compact addressing information, while none of `ranges/{tile_indptr,value_id,row_start,row_count}` is needed by this route. The complete-tile batch reader continues to combine multiple such intervals into its coordinated Zarr selections.

**Compact complete-tile addressing**

`_PointsCacheReader._load_runtime_indexes()` already loads the manifest's bucket identity, bucket-local tile index, and `n_points`. Derive offsets once from those counts, independently for each `(level, bucket_id)` and in bucket-local tile order. For example:

```text
Bucket-local tile index:  0       1       2
Manifest n_points:        3       5       2
Derived tile_offset:  [0, 3,      8,     10]

Tile 1 occupies this bucket's point rows [3:8).
```

These offsets use every tile in the bucket, not just the current viewport or CPU-residency misses. They are compact: one boundary per tile, rather than one record per tile/value range. Validate them against the persisted bucket offsets when that bucket is first opened, rather than opening every bucket during startup merely to obtain `tile_offset`. This is an in-memory addressing change; it does not require a new persisted cache array.

**Persisted arrays versus the resident lookup object**

Removing sparse indexes from the viewer does not mean deleting their stored arrays or necessarily deleting `_BucketLookupIndex` everywhere:

- **Normal viewer startup and reads:** neither physical route loads or retains `_BucketLookupIndex`. All-values reads use compact offsets; selected-value reads use the value-major cache and the active selection's `value_tiles` records.
- **Construction and validation:** persisted bucket sparse ranges remain consumed, for example by `_iter_compact_bucket_range_batches()` during catalog generation and validation. These consumers read the stored arrays in batches; their need for those arrays does not imply a dependency on the resident `_BucketLookupIndex` object.
- **Explicit diagnostic/reference reads:** at the Slice 10 checkpoint, `read_tile(..., value_ids=...)` and bucket-level selected-value display APIs still support sparse tile-major subset reads. The tile-major/value-major equivalence tests use that path as their reference. This path needs an explicitly loaded sparse lookup, outside the normal viewer session, and must never become an implicit viewer fallback. Slice 10c replaces these reads with complete tile-major reads and independent in-memory filtering, then removes `_BucketLookupIndex`; deletion everywhere is not a Slice 10 exit requirement. Optional Slice 17 separately evaluates removing the persisted ranges.

**Production changes**

1. Split bucket addressing into:
   - compact complete-tile offsets; and
   - large sparse selected-value ranges excluded from normal viewer reads, while remaining available to construction, validation, and explicit diagnostic/reference reads as described above.
2. Keep compact manifest, catalog value pointers, per-level sidecar pointers, totals, and the derived complete-tile addressing resident. Refactor `resolve_complete_tile_interval()` and its display-reader callers to use the compact offsets independently of `_BucketLookupIndex`. Preserve `_read_complete_tile_major_requests()`'s one-batch-per-bucket payload selections, logical output order, and cancellation checkpoints; do not replace them with per-tile Zarr reads. Do not treat all five current bucket lookup arrays as one indivisible load unit.
3. Remove the unconditional all-level `project_bucket_lookup_index_bytes()` and `load_bucket_lookup_indexes()` sequence from `_TiledPointsCacheWorker.start()`.
4. At the Slice 10 checkpoint, route every proper subset through its selected level's sidecar and every all-values request through compact complete-tile addressing. Neither branch may call `load_bucket_lookup_indexes()` or `_BucketReader.load_lookup_index()`. Slice 11 may later add complete tile-major reads plus in-memory filtering as a second proper-subset route, but it must use the same compact addressing and remain independent of sparse ranges.
5. Remove `max_bucket_lookup_bytes` from `TiledPointsApplicationSettings`, `_CacheSessionSettings`, adapter wiring, startup progress, diagnostics, benchmarks, and tests. This is a direct removal rather than a compatibility migration because the viewer no longer has a bucket sparse-index allocation to bound.
6. Keep open bucket-reader metadata and point-payload access independent from sparse-index residency. Opening a tile-major bucket for an all-values payload must not load its `ranges` arrays.
7. Preserve persisted `ranges/row_start` and the other sparse-range arrays in this slice for cache construction, catalog generation, and publication validation. Any schema simplification is later, separate work.
8. Replace startup progress/status that assumes a complete index load with compact-metadata and sidecar-ready diagnostics.

**What stays unchanged**

- The active selection's in-memory `value_tiles` records and counts remain necessary for planning and value-major addressing. They are distinct from the bucket `ranges` arrays being removed from viewer residency; `max_selected_value_index_bytes` remains meaningful.
- Keep point-level tile-major `value_id` on disk and read it alongside `location` for all-values payloads. Removing resident `ranges/value_id` does not remove point-level colour IDs.
- Keep the cache schema and persisted sparse ranges for construction, catalog generation, and validation. This slice changes viewer residency, not the stored format.
- LOD selection, the logical tile payload contract, decoded CPU tile retention, worker packing, and the visual/VBO path are unchanged. Adaptive physical routing remains Slice 11 work.

**Focused tests**

- Session startup reaches ready without projecting or loading sparse ranges.
- Proper-subset requests at every level and all-values requests leave sparse resident bytes and sparse-index load counts at zero.
- Patch `project_bucket_lookup_index_bytes()`, `load_bucket_lookup_indexes()`, and bucket `load_lookup_index()` to fail and prove normal viewer startup and reads do not touch them.
- Repeated Exact, Bridge, and Spatial level/view changes never create a sparse lookup index.
- All-values tile-major reads remain correct using compact complete-tile addressing alone.
- Derived offsets reset at each level/bucket boundary and use complete bucket counts even when only later or disjoint tiles are requested. Opening a bucket rejects a mismatch with its persisted offsets rather than silently reading the wrong point rows.
- Preserve complete-tile batching, output order, and cancellation behavior without requiring sparse-index priming in those tests.
- Removing `max_bucket_lookup_bytes` leaves no constructor, settings, adapter, diagnostic, or test compatibility alias.
- Construction and independent staged validation still consume the persisted ranges correctly outside the viewer runtime.
- Keep explicit tile-major subset reference reads separate from the viewer no-sparse-index checks. Any lookup priming retained for a diagnostic/reference reader must not prime the reader/session being tested for zero sparse-index residency.

**Benchmark evidence**

Report startup metadata time, time to ready, compact resident bytes, per-level sidecar-pointer bytes, open bucket readers, sparse resident bytes, sparse-index load count, and peak RSS. Account for compact complete-tile offsets separately from sparse ranges so retaining the necessary offsets is not reported as a sparse-index allocation. Sparse resident bytes and load count must remain zero across Exact, Bridge, Spatial, all-values, and repeated viewport traces, and the previous 568.4-MiB eager allocation must disappear.

Also measure first all-values payload latency and warm repeated reads: lazy bucket opening and offset validation still have a cost when a bucket is first needed. Report that work separately from time to ready rather than treating deferred work as eliminated. Verify unchanged logical payloads and physical batching; do not claim a direct draw-time or zoom-stutter fix from this startup/memory slice.

**Implementation and measured results — 7 September 2026**

The viewer now uses immutable `_BucketTileIndex` offsets derived from the complete manifest, separately for each level and bucket. First complete-tile use checks the offsets, tile coordinates, and counts against storage before accepting them; subsequent reads reuse the accepted addressing without pointer IO. Serialized bucket IDs may have gaps because empty hash buckets are omitted, so validation counts distinct buckets rather than treating `bucket_count` as an upper bound on their IDs.

Session startup no longer projects or loads sparse indexes. `max_bucket_lookup_bytes`, the bucket-index loading state, progress signals, and their application/widget wiring are removed without aliases. Readiness reports compact resident-index bytes. Persisted ranges and explicitly primed diagnostic subset reads remain supported; equivalence tests use separate reference readers. Neither normal physical route loads these diagnostic indexes.

Measurements used the existing completed cache, generation `939bdb3e-d137-49d5-9d3e-0779c67e4156`; no cache rebuild was needed. Python imports are excluded from startup latency. Filesystem caches were not flushed.

| Startup/residency measurement | Result |
|---|---:|
| Controlled reference restoring the previous eager project/load sequence | 8,212.5 ms; 108 open buckets; 568.4 MiB lookup arrays |
| Normal compact-reader startup in the viewport trace | 156.3 ms; 0 open buckets |
| Actual worker-session start → GUI `ready` signal | 145.9 ms, including 145.6 ms metadata loading |
| Total compact resident NumPy indexes | 1,328,400 bytes / 1.267 MiB |
| Complete-tile offsets, included above | 138,056 bytes / 0.132 MiB |
| All per-level value-major point pointers, included above | 368,856 bytes / 0.352 MiB |
| Sparse-index residency and loaded-index count in normal reads | 0 bytes; 0 indexes |
| Session-probe peak RSS, including imports and repeated reads | 321.2 MiB |

Separate instrumented cache-to-canvas runs, without physical canvas drawing:

| Request | Level; tiles; points | First worker snapshot | CPU-resident repeat | Worker-phase peak RSS |
|---|---|---:|---:|---:|
| AAMP | Exact; 4,453; 60,512 | 96.3 ms | 42.1 ms | 325.9 MiB |
| ADAMTS1 | Bridge; 5,853; 92,499 | 133.6 ms | 53.8 ms | 324.5 MiB |
| All values | Spatial 8; 1; 100,000 | 156.2 ms | 1.2 ms | 322.4 MiB |

Both selected-value runs kept zero tile-major bucket readers open. The all-values run opened one bucket and spent **74.2 ms** of its first snapshot on lazy bucket setup/address validation, followed by **80.3 ms** in the batched payload reader. This cost is deferred, not eliminated. A later filesystem-warm session probe delivered its first all-values snapshot in 27.2 ms and subsequent CPU-resident snapshots in approximately 1.1 ms. Explicitly rereading the same physical payload, bypassing CPU residency, took a 10.3-ms median after the first read. These are different cache states, not contradictory estimates of one fixed first-read cost.

The planning benchmark covered two selections, full and partial viewports, and empty, partial, and complete CPU residency, with five repeats per case. All 60 packed-batch hashes matched the earlier reference and the controlled eager-loading reference. Sparse-index bytes/count and open bucket readers remained zero throughout the normal selected-value traces. On the repeated comparison, cold-worker medians for full-extent one-/two-value requests were 96.8/147.4 ms versus 97.8/142.8 ms with eager priming restored: comparable payload-processing costs, not a claim of a rendering speedup.

The final focused run passed **215 tests** covering startup without sparse projection/loading, repeated Exact/Bridge/Spatial reads, disjoint tiles across multiple buckets, nonzero offsets, corrupt addressing rejected before payload IO, one-time validation, batching, cancellation, and separate reference readers. This includes construction, staged validation, exhaustive validation, and affected application/widget tests. Ruff checks and formatting checks pass on the changed Python files. The full suite and physical OpenGL drawing were not rerun for this residency-only change.

Evidence files: `/private/tmp/napari-harpy-slice10-session-startup.json`, `/private/tmp/napari-harpy-slice10-aamp-isolated.json`, `/private/tmp/napari-harpy-slice10-bridge.json`, `/private/tmp/napari-harpy-slice10-all-values.json`, `/private/tmp/napari-harpy-slice10-viewport-planning.json`, `/private/tmp/napari-harpy-slice10-viewport-planning-repeat.json`, and `/private/tmp/napari-harpy-slice10-eager-reference-planning.json`. The eager reference deliberately primes diagnostic lookups outside the normal viewer runtime to isolate the removed startup policy.

**Exit condition**

Large sparse ranges are absent from normal viewer startup and reads; persisted ranges remain available for construction, validation, and explicit diagnostic/reference consumers outside that runtime. Startup does not eagerly open every bucket to prime lookup indexes, and both viewer physical read routes work without sparse-index loading. Complete-tile reads retain compact addressing and their existing batched payload behavior. This does not require deleting `_BucketLookupIndex` or the explicitly primed reference path from the entire codebase.

### Slice 10b — Self-contained complete-tile addressing in `_TileDescriptor`

**Status: Implemented**

This is a contract-simplification follow-up to implemented Slice 10, scheduled before adaptive physical routing in Slice 11. It keeps the zero-sparse-index viewer architecture but moves each complete tile's point-row start into its immutable descriptor. The new field replaces the separate `_BucketTileIndex` representation; it must not create a second retained copy of the same derived addressing. Later slice numbers remain unchanged.

**Motivation and descriptor contract**

`_TileDescriptor` already identifies a finalized tile's physical bucket and its tile index within that bucket. Before this slice, finding the tile's point-row interval additionally required `_BucketTileIndex.tile_offset`. The descriptor is now self-contained:

```python
@dataclass(frozen=True)
class _TileDescriptor:
    level: int
    bucket_id: int
    bucket_tile_index: int
    bucket_row_start: int
    tile_x: int
    tile_y: int
    n_points: int
```

- Keep `bucket_tile_index` in Python and the persisted manifest: the zero-based index among all nonempty tiles in this bucket, ordered by `(tile_y, tile_x)`. It indexes persisted tile-coordinate arrays and sparse-range pointers where those remain in use. The separate `bucket_row_start` field makes the distinction from a point-row address explicit; no alternative field name or compatibility alias is needed.
- Introduce required `bucket_row_start`: the zero-based point-row address in the tile-major bucket's aligned `location`, point-level `value_id`, and `point_id` arrays. Prefer this name over `bucket_tile_offset` to make the unit explicit. It is not a byte offset, spatial origin, or value-major row address.
- Keep `n_points` as the complete tile's stored point count. The half-open interval is `[bucket_row_start, bucket_row_start + n_points)`; do not store a redundant stop field.
- Require a valid nonnegative integer start and a positive count whose sum remains within the supported signed-64-bit row domain. The new field is never optional and has no placeholder default.

For a bucket whose tiles contain `[3, 5, 2]` points, descriptors have tile indexes `[0, 1, 2]` and row starts `[0, 3, 8]`. The second descriptor therefore directly describes rows `[3:8)`. These are complete-bucket addresses, independent of viewport, active values, LOD selection, and CPU-residency misses within that level.

**Production changes**

1. Update every descriptor producer, not only the viewer reader. `_BucketWriter._reconcile_result()` takes starts from its already available planned offsets; independent bucket validation obtains them from reopened stored offsets. `_PointsCacheReader._load_runtime_indexes()` and `_read_manifest_inventory()` derive starts from complete manifest counts, resetting running totals for each `(level, bucket_id)`. Compute the start before constructing each frozen descriptor, rather than creating incomplete descriptors and mutating or replacing them afterward. Preserve support for gaps in serialized bucket IDs and reject inconsistent bucket-local tile-index order.
2. Retain `_descriptors` in manifest order and `_descriptors_by_bucket` as grouped references to those same objects. Remove `_BucketTileIndex`, `_bucket_tile_indexes`, and their installed offset-array state. Temporary arrays used for storage comparison are acceptable, but do not retain a parallel bucket offset array after validation. `_BucketReaderCache` remains a cache of opened reader objects and is not removed by this refactor.
3. Replace `set_tile_index()` with validation and installation of the complete immutable descriptor tuple for that bucket. Before accepting it, check matching bucket identity, contiguous tile indexes, tile coordinates, starts beginning at zero, adjacent intervals without gaps or overlaps, matching tile/point totals, and agreement with persisted `tile_offset`. Publish the accepted tuple only after all checks succeed. Subsequent reuse performs no pointer IO; failed validation leaves no accepted addressing, and reader closure releases the accepted state.
4. Make `resolve_complete_tile_interval()` use `descriptor.bucket_row_start` and `descriptor.n_points` after checking that the requested descriptor belongs to the bucket's accepted descriptor set. Keep that check constant-time through its tile index; do not scan the bucket or recompute prefixes per request. A boolean saying that some descriptors were validated must not authorize arbitrary replacement descriptors. Preserve `_read_complete_tile_major_requests()` batching, order, cancellation, and lazy bucket opening.
5. Keep construction, staged validation, exhaustive validation, and explicitly primed diagnostic/reference reads independent where they currently compare persisted structures. In particular, do not replace a stored-offset read with the same manifest-derived start on both sides of an equivalence check. Update these consumers and their test fixtures to the complete descriptor contract. `_BucketLookupIndex` and explicitly requested sparse subset reads remain available outside normal viewer sessions at this checkpoint; Slice 10c removes that in-memory lookup contract, while optional Slice 17 separately evaluates removing the persisted ranges.
6. Update descriptor-related docstrings, examples, benchmarks, and `CACHE_FORMAT.md`. Distinguish retained NumPy-index bytes from Python descriptor storage: removing the NumPy offset array does not make row-address storage free. Remove obsolete array-specific accounting and report the remaining array bytes, descriptor count, and process RSS with their scopes stated clearly.

**Persisted schema boundary**

This slice changes Python objects, not the stored cache format. Keep the existing `manifest/bucket_tile_index` array and bucket `tile_offset` arrays unchanged. The Python field and persisted column both use `bucket_tile_index`. Derive `bucket_row_start` when reopening the manifest; do not introduce a new manifest offset column or require a cache rebuild.

**Focused tests**

- Descriptors produced by construction, manifest reopening, and independent bucket validation agree on tile index, row start, and complete point count.
- Starts reset independently across levels and buckets, including non-dense bucket IDs, and remain correct when reading only later or disjoint tiles. Valid manifest grouping needs no additional sort.
- Invalid starts, overflow, duplicate or reversed tile indexes, gaps/overlaps, incorrect coordinates/totals, and mismatches with stored offsets fail before payload IO. A mismatched request descriptor cannot borrow another descriptor's validation.
- First-use validation is atomic and runs once for the accepted tuple; warm reads do not reload tile pointers, and failure/closure retain no accepted stale addressing.
- All-values reads and value-major subset reads preserve ordered logical payloads across Exact, Bridge, and Spatial levels. Patch sparse-index projection/loading to fail in normal viewer tests; keep independently primed reference readers separate.
- Construction and both validation modes still exercise the persisted arrays. Complete-tile batching, cancellation, startup readiness, and deterministic cleanup remain unchanged.

**Performance checks and limits**

The additional Python integer is **per stored tile, not per point**. Runtime descriptors are constructed during reader initialization, not recreated for every viewport or frame. The point arrays, packed vertex layout, and VBO contents remain unchanged; storing the row start on the descriptor introduces no additional per-point processing. Python integer objects have more storage overhead than entries in a compact `uint64` array, but that does not imply slower scalar access.

A local prototype on 7 September 2026 compared the current representation with the proposed required row-start field, using **17,149 tile descriptors across 108 buckets**:

- **Net additional retained allocation:** 506,687 bytes, approximately **0.48 MiB**, accounting for removal of the separate bucket-offset representation. The comparison includes newly allocated descriptors, grouping containers, and addressing; existing shared scalar fields are excluded equally from both sides. This is not a measurement of total process RSS.
- **Metadata-object construction:** median **19.7 → 21.1 ms**, approximately **1.5 ms additional initialization work**. This measures object construction and descriptor validation, not complete reader startup or Zarr access.
- **Isolated scalar row-address lookups across all 17,149 tiles:** median **3.38 → 1.18 ms**. Direct access avoids the current extraction and conversion of two NumPy scalars, but this microbenchmark excludes the full reader call and validation path.

These are **prototype measurements, not end-to-end acceptance results**. They support treating the memory increase as a small metadata trade-off at this tile count, not as evidence of an expected rendering regression or a promised overall speedup. The prototype did not perform bucket first-use validation, point payload IO, or physical drawing. Evidence: `/private/tmp/napari_harpy_descriptor_offset_probe.py` and `/private/tmp/napari-harpy-descriptor-offset-probe.json`.

After implementation, compare complete metadata startup, first-use bucket validation, first complete-tile read, warm reads, and repeated viewport request latency with Slice 10. Verify unchanged output and zero sparse-index residency. Check RSS and startup cost, including how the descriptor overhead scales with stored tile count, without adding a new memory-budget mechanism or changing LOD, CPU residency, packing, or GPU resources. This remains a clarity improvement, not an assumed memory or rendering optimization. No cache reconstruction benchmark is required because the persisted schema is unchanged.

**Implementation and verification — 7 September 2026**

All descriptor producers now supply the required row start, and the Python field uses `bucket_tile_index`, matching the stored column. The viewer no longer retains `_BucketTileIndex` or a parallel derived offset array. Each lazily opened bucket accepts its complete descriptor tuple only after checking tile indexes, coordinates, contiguous point intervals, totals, and persisted offsets. Warm requests use the tile index to verify the requested descriptor against that accepted tuple, then read its start and count directly. Construction and explicitly primed diagnostic/reference reads still compare against independently read stored offsets. The persisted schema is unchanged; the existing cache was used without rebuilding.

A read-only before/after comparison used 17,149 descriptors across 108 buckets, seven fresh reader sessions per version, and a full-extent all-values request selecting Spatial level 8 with 100,000 points. Operating-system caches were not flushed. Bucket setup includes lazy store opening and the first descriptor/address validation; physical reads below bypass CPU tile residency. Medians:

| Measurement | Before | After |
|---|---:|---:|
| Complete metadata startup | 73.18 ms | 74.81 ms |
| First-use bucket setup/validation | 10.36 ms | 9.59 ms |
| First complete-tile read, including that setup | 20.30 ms | 19.25 ms |
| Repeated physical complete-tile read | 9.45 ms | 9.17 ms |
| Repeated worker request with full CPU residency | 0.955 ms | 0.936 ms |

Retained NumPy-index bytes changed from **1,328,400 to 1,190,344**, removing the **138,056-byte** derived offset arrays. These numbers exclude Python descriptors and containers. Median process RSS after startup was **323.0 → 325.9 MiB**, while whole-process peak RSS was **328.6 → 325.9 MiB** in the separate runs. RSS includes allocator and process variability; it does not isolate descriptor overhead or establish a memory saving. The prototype above remains a separate estimate of per-tile allocation cost.

The selected-value trace covered one/two selected values, full/20%-extent viewports, and empty/partial/full CPU residency, with three repeats per case. **All 36 packed-batch hashes matched the baseline**; all-values payload and packed-batch hashes also matched. Full-extent cold-worker medians were **107.14 → 97.11 ms** and **152.79 → 145.62 ms**; full-residency medians were **41.07 → 40.55 ms** and **56.75 → 56.33 ms**. No measured read-path slowdown was observed, but these runs do not establish a speedup attributable to the descriptor refactor. Sparse-index bytes/count remained zero; startup and selected-value traces opened no tile-major bucket readers.

The final focused run passed **307 tests**, covering descriptor bounds, writer/manifest/independent-validator agreement, non-dense bucket IDs, rejection of invalid addressing, one-time atomic validation, mismatched request descriptors, cleanup, all-level logical equivalence, construction, staged/exhaustive validation, batching, cancellation, and worker integration. Ruff lint and formatting checks pass. The full test suite and physical OpenGL drawing were not rerun; point payloads, packing, and GPU code are unchanged.

Evidence: `/private/tmp/napari_harpy_complete_address_benchmark.py`, `/private/tmp/napari-harpy-slice10b-before.json`, `/private/tmp/napari-harpy-slice10b-after.json`, `/private/tmp/napari-harpy-slice10b-planning-before.json`, and `/private/tmp/napari-harpy-slice10b-planning-after.json`. The selected-value trace uses `scripts/benchmark_tiled_points_viewport_planning.py`.

**Exit condition**

Every finalized `_TileDescriptor` carries an explicit tile index and complete point-row start. Normal complete-tile reads use that descriptor after one-time bucket validation, with no separate `_BucketTileIndex` or retained derived offset array. Existing caches, independent validation, diagnostic reference reads, physical batching, and the viewer's zero-sparse-index contract remain intact.

### Slice 10c — Remove the complete-tile fallback and retire `_BucketLookupIndex`

**Status: Implemented**

This is a bounded contract-simplification follow-up to Slice 10b, scheduled before adaptive routing in Slice 11. First remove the alternate complete-tile lookup path, then replace the remaining diagnostic sparse-subset consumers and remove the resident lookup machinery. Later slice numbers remain unchanged. This changes Python reader contracts, not the published cache format, and does not depend on optional Slice 17.

**Current code and scope**

Before this slice, `_BucketReader.resolve_complete_tile_interval()` accepted two initialization modes: an installed, validated descriptor tuple, or an explicitly loaded `_BucketLookupIndex` whose `tile_offset` supplied the interval. Normal viewer complete-tile reads already used the first mode through `_PointsCacheReader._get_bucket_reader_for_complete_display()`. Standalone bucket callers and tests retained the second mode.

Removing that fallback alone does not make `_BucketLookupIndex` unused. `resolve_selected_tile_intervals()` still consumes its sparse ranges through bucket display APIs and `read_tile(..., value_ids=...)`. Value-major equivalence tests use these explicitly primed reads as their independent tile-major reference; diagnostic benchmarks also reference the APIs. These consumers must be adapted before deleting the object.

Keep the distinction between the **resident Python lookup** and the **persisted bucket arrays** explicit. Construction, catalog generation, staged validation, and optional exhaustive validation consume stored ranges directly, including through `_iter_compact_bucket_range_batches()`. They do not require `_BucketLookupIndex` and remain unchanged by this slice.

**Implementation sequence**

1. **Require accepted descriptors for every complete-tile display read.** Remove the `_lookup_index_or_raise()` fallback from `resolve_complete_tile_interval()`. If no descriptor tuple has been installed, raise a clear error before payload IO. Adapt standalone callers and fixtures to call `set_tile_descriptors()` with the complete bucket tuple. Preserve first-use comparison with persisted offsets and coordinates, atomic installation, constant-time request-descriptor checks, warm tuple reuse, and cleanup. Do not replace the fallback with implicit index loading or unvalidated descriptor addressing. Construction's independent stored-offset reads remain separate and unchanged.
2. **Replace diagnostic subset reads before removing their implementation.** Read complete tile-major `location` and point-level `value_id`, then apply the same in-memory membership mask to both arrays. The singleton `read_tile(..., value_ids=...)` convenience API can retain its selection argument with this explicit diagnostic contract; it must no longer require lookup priming or resolve sparse ranges. Preserve canonical point order, dtypes, immutable returned payloads, and `None` for a missing tile or a tile with no matching points. Multi-tile diagnostic consumers must retain coordinated bucket reads rather than introduce per-tile payload IO loops.
3. **Keep equivalence references independent.** Replace lookup-primed test references with complete tile-major reads and an explicit point-level membership filter. Expected results must come from actual tile-major point arrays, not value-major reads or reconstructed IDs from the selected-value catalog. This preserves an independent check of both locations and value IDs. Adapt bucket-reader and writer tests that exercised selected display reads; retain direct persisted-range validation coverage instead of keeping an obsolete display API solely for tests.
4. **Make bucket display APIs complete-tile only, rename them, and delete retired machinery.** Rename `_BucketReader.read_display_payload()` to `read_complete_display_payload()` and `read_display_payloads()` to `read_complete_display_payloads()`. Remove their per-request `selected_value_ids` arguments, homogeneous-mode branching, `resolve_selected_tile_intervals()`, `_ResolvedSelectedValueRange`, and selected-ID synthesis once their consumers are migrated. Remove `_BucketLookupIndex`, its reader state, load/release helpers, projection and resident-byte accounting, cache-wide priming/progress/rollback code, and tests of those retired contracts. Remove helpers only after checking remaining consumers. Preserve `_BucketReaderCache` as the lazy cache of opened readers; preserve descriptor addressing, shared exact-row selection, batch output partitioning, and point-ID-free display reads. Do not retain the old method names as compatibility aliases or add a hidden sparse-subset fallback.
5. **Update tooling and documentation together.** Update every caller, test, monkeypatch, and benchmark hook to the new complete-display method names, including the viewer's `_read_complete_tile_major_requests()` and diagnostic `read_tile()`. Audit standalone bucket/exact/acceptance benchmarks, reader fixtures, worker test doubles, and normal-viewer sparse-access guards. Retarget the cache-to-canvas bucket-batch timing hook to `read_complete_display_payloads()`; remove its hook for `resolve_selected_tile_intervals()` and obsolete lookup metrics so renamed or deleted methods do not break instrumentation. Relabel diagnostic measurements as complete-tile reads plus filtering; do not compare them as if they still measured sparse-range IO. Update affected docstrings and runtime explanations in `CACHE_FORMAT.md` without changing its persisted schema description.
6. **Document the cache reader's responsibilities and ownership.** Expand the `_PointsCacheReader` class docstring according to the requirements below. Define catalog arrays, include a compact reader/storage relationship scheme, and explain metadata residency versus payload IO. Describe the actual complete-display APIs and routing after this cleanup, without references to roadmap slices, benchmark datasets, or planned-but-unimplemented behavior in source documentation.
7. **Review overlapping tests without losing boundary coverage.** Compare `test_compact_tile_addressing.py` with the existing model, reader, bucket-reader, and viewport-read tests while migrating the retired APIs. Consolidate genuinely overlapping broad read/round-trip coverage where appropriate, but retain the distinct descriptor-validation and lifecycle guarantees described below. Do not remove the module wholesale on the assumption that successful reads elsewhere cover those guarantees.

In the new names, **complete** means every point in each requested tile, without value filtering; it does not mean every tile in the cache. **Display** means `location` and point-level `value_id`, excluding `point_id`, unlike construction payloads. Docstrings must explicitly identify tile-major storage. The singular method remains a one-tile wrapper around the plural batched reader, which remains in the normal viewer's all-values path. These names describe the physical read, not a particular viewer route: Slice 11's future `tile_major_filter` route will also consume complete display payloads before filtering them in memory.

**Required `_PointsCacheReader` docstring clarification**

Define **catalog arrays** as cache-wide lookup metadata, distinct from point payloads. Give concrete examples rather than relying on the word "compact": `manifest/*` describes existing tiles, physical buckets, logical tile coordinates, and point counts; `values/n_points` stores per-value totals; `value_tiles/*` maps values to manifest tiles and their counts. This metadata supports both physical point orderings. The catalog is not another name for tile-major storage, and "catalog arrays" does not imply that every value/tile record is loaded at startup.

Include a small scheme connecting the coordinating reader, its lower-level readers, and the arrays they access, along these lines:

```text
_PointsCacheReader — coordinates reads for one cache generation
    |
    +-- _CacheRootReader
    |     cache-wide metadata and value-major array handles
    |
    +-- _BucketReaderCache
    |     +-- _BucketReader per opened bucket
    |           tile_major/level_N/bucket-....zarr
    |           complete display reads: location + value_id
    |
    +-- _ValueMajorLocationReader per level
          value_major/level_N/location
          selected value/tile intervals
```

Explain that `_ValueMajorLocationReader` wraps an array handle owned and validated by `_CacheRootReader`; it does not open an additional store. Opening an array handle is not reading its point payload. `_BucketReaderCache` retains lazily opened bucket readers and their metadata, not decoded chunks or point payloads. CPU tile residency is owned outside `_PointsCacheReader`. The cache reader supplies each bucket reader with its existing per-bucket descriptor tuple for complete-tile validation and addressing, without copying those descriptors.

Document **descriptor construction, sharing, and validation** here rather than in the manifest-format explanation. Each manifest row becomes one immutable `_TileDescriptor`, retained in manifest order in `_descriptors`; `_descriptors_by_bucket` groups references to those same objects, and each bucket reader receives the existing tuple for its bucket. Explain that `bucket_tile_index` matches the persisted manifest column, while `bucket_row_start` is derived once from complete preceding tile counts within each `(level, bucket_id)`, including tiles outside the viewport. The reader retains no parallel derived offset array. On first complete-tile use of an opened bucket, validate its full descriptor tuple against stored `tile_offset`, tile coordinates, and totals before accepting it or reading display payloads. Distinguish that atomic first-use check from warm reuse of the accepted tuple, and explain release of the retained references on closure.

Distinguish three lifecycle stages:

- **Reader startup:** load the complete manifest, compact value totals and index pointers, and per-level value-major point pointers; derive complete-tile descriptors once. Construct value-major array wrappers without decoding location payloads, and do not eagerly open tile-major buckets. Describe routine opening/layout checks separately from independent staged or exhaustive validation.
- **Selection changes:** load the requested values' `value_tiles` records through `load_selected_value_index()`. The caller retains the returned index and reuses it for planning and reads; neither the entire value/tile catalog nor a fresh selection index is loaded for every viewport.
- **Payload requests:** `read_planned_tiles()` dispatches all-values requests to batched complete tile-major reads and proper subsets to value-major interval reads. Explain that `_PointsCacheReader` assembles the same ordered logical tile results from either layout, including regrouping value-major locations and reconstructing their aligned value IDs. Diagnostic `read_tile(..., value_ids=...)` remains distinct: complete tile-major reads followed by in-memory filtering.

Keep this a reader-level explanation, not a duplicate of the full persisted-format document. `CACHE_FORMAT.md` should explain the stored relationship between manifest tile indexes, counts, and bucket `tile_offset`, including the per-bucket prefix example; Python descriptor objects, tuple sharing, and validation lifecycle belong in source reader documentation. Use the implemented method names, make resource ownership and release on reader closure clear, and remove descriptions of retired lookup priming. The future adaptive route remains separate implementation work; source docstrings must not present it as already available.

**Boundaries and non-goals**

- Keep bucket `tile_offset`, tile-coordinate arrays, and `ranges/{tile_indptr,value_id,row_start,row_count}` unchanged on disk. Keep the manifest, `value_tiles`, value-major pointers, and both physical point orderings. No schema version change, compatibility layer, cache rebuild, or in-place cache modification is required.
- Preserve construction and both validation modes, including their independent stored-offset and range-to-catalog comparisons. Removing runtime lookup loading must not weaken these checks or replace them with self-comparisons.
- Do not add `tile_major_filter` to normal viewer routing here. That production route, its bounded transient allocations, cost estimator, cancellation, and route selection remain Slice 11 work. The viewer continues using value-major reads for proper subsets and complete tile-major reads for all values.
- This is a maintenance simplification, not an expected rendering-speed or viewer-RSS improvement: normal viewer reads already avoid this lookup. Diagnostic filtering can read more point rows, decode point-level `value_id`, and require more temporary memory than sparse subset reads. Keep diagnostic work bounded and report that cost separately from unchanged viewer performance.

**Focused tests and verification**

- A complete-tile display read without installed descriptors fails before payload IO; installing the correct tuple enables it. Existing first-use validation, mismatch rejection, warm no-pointer-IO behavior, and closure tests remain effective.
- Complete bucket batches preserve one coordinated location/value-ID selection, request order, shared result partitioning, and the guarantee that display never reads point IDs. Replace retired mixed-mode tests with the complete-only API contract.
- Diagnostic filtering and value-major reads return identical logical tiles across Exact, Bridge, and Spatial levels, including multiple values, partial/missing-tile requests, absent values, and tiles whose filtered output is empty. References must use tile-major point-level IDs and stay independent of the value-major implementation.
- Normal viewer startup, all-values reads, and value-major subset reads still avoid persisted sparse-range payload access. Replace monkeypatches targeting deleted loader methods with guards at the actual range-array read boundary; mere absence of an API is not the behavioral proof.
- Focused construction, staged-validation, and exhaustive-validation tests still exercise the stored ranges and reject their existing corruption cases. Reader batching, cancellation, worker readiness, and resource cleanup remain unchanged.
- Smoke-test affected benchmark hooks and scripts. Compare representative cold/warm viewer outputs and timings with the existing cache; measure diagnostic filtering time and temporary-memory cost separately. Do not claim a rendering improvement from deleting code that was already outside the viewer path.
- Review the class docstring and scheme against reader startup, selection-index loading, physical dispatch, bucket setup, and closure. They must distinguish resident metadata, open array/store handles, and decoded payloads, without claiming that all catalog records are eagerly resident or that bucket readers serve value-major payloads.

**Test-consolidation scope**

Basic descriptor field bounds and interval overflow are already covered in `test_models.py`; successful reads and ordering are covered in `test_reader.py`; disjoint complete-tile reads across levels and buckets are covered in `test_value_major_viewport_reads.py`. This overlaps with some successful-read assertions in `test_compact_tile_addressing.py`, but does not make its runtime validation tests redundant.

- Preserve focused coverage of first-use validation followed by warm reuse of the identical descriptor tuple, no repeated pointer IO, and release on closure.
- Preserve rejection of manifest/bucket disagreement before payload IO, atomic installation with no accepted state after failure, retry with valid descriptors, and rejection of mismatched requests that would otherwise borrow prior validation. Model-level field checks and independent construction reads do not exercise these installed-state contracts.
- Preserve agreement between runtime descriptors, manifest inventory, and independent bucket validation, plus row-start reset across levels and nonconsecutive bucket IDs without opening buckets. Keep the behavioral no-sparse-access guards through selection and viewport changes.
- Consider consolidating overlapping broad read/round-trip cases only after comparing their actual assertions and scenarios. Retain requested-tile ordering, skipped-tile offsets, all-level coverage, selection transitions, independent payload references, and IO guards in the surviving tests. Similar inputs or test names alone are not evidence of duplication.
- Retire the sparse-lookup-specific parts of `test_primed_display_reads_do_not_reread_bucket_lookup_arrays()` together with the removed priming API. Preserve its warm complete-read guarantee in `test_complete_tile_descriptors_are_validated_once_reused_and_released()` or an equally focused replacement; do not keep an obsolete API merely to preserve the old test.

For any consolidated test, identify which surviving test retains its distinct guarantees. This is a bounded review alongside the reader-contract migration, not a requirement to reduce test counts, delete `test_compact_tile_addressing.py`, or reorganize unrelated tests.

**Implementation and verification (2026-09-08)**

- Complete display reads now require an accepted descriptor tuple. The bucket APIs are `read_complete_display_payload()` and `read_complete_display_payloads()`, accepting a descriptor or a tuple of descriptors, with no per-tile selection argument or compatibility aliases. Complete tiles are nonempty by contract, so these APIs return `_PointDisplayPayload` rather than optional payloads.
- Diagnostic `read_tile(..., value_ids=...)` reads complete tile-major point arrays and filters both with one membership mask. Missing tiles and empty filtered results still return `None`. Independent multi-tile references retain bucket batching and filter actual point-level IDs. The resident lookup object, fallback, sparse-subset resolver, ID synthesis, priming/projection/rollback code, and lookup metrics have been removed; persisted arrays and their construction/validation consumers are unchanged.
- Migrated benchmark hooks, standalone bucket/Exact diagnostics, acceptance measurements, selected-index reports, and worker test doubles. Diagnostic timings and decode-amplification estimates now describe complete reads plus filtering. Added `test_benchmark_readers.py` to exercise both physical-route timing hooks and the affected diagnostic/reporting helpers on small caches.
- **Test consolidation:** kept `test_compact_tile_addressing.py` and its distinct first-use validation, atomic rejection/retry, tuple sharing, closure, and independent-address checks. Retired `test_primed_display_reads_do_not_reread_bucket_lookup_arrays()`; its warm complete-read guarantee remains in `test_complete_tile_descriptors_are_validated_once_reused_and_released()`, with persisted-range access guards retained separately. Replaced obsolete mixed-mode/selected-range bucket tests with complete-batch validation, disjoint selection, shared-allocation, and no-point-ID tests. Diagnostic filtering and invalid selected IDs are tested at the cache-reader boundary. Replaced the impossible complete-batch `None` injection with a real array-boundary truncated-payload rejection test.
- **198 focused test cases passed** across bucket/cache readers, addressing, value-major viewport reads, reader lifetime, Exact construction, staged and exhaustive validation, benchmark helpers, and worker sessions. Changed Python files pass Ruff lint/format checks; `git diff --check` passes. Dependency deprecation warnings remain. No full repository suite or real-canvas/GPU timing run was required for this reader-only cleanup.

Existing-cache worker comparisons used five repeats per selected-value case, at a 100,000-point budget. Cold means empty CPU tile residency, not a flushed filesystem cache. The before/after render-batch SHA-256 hashes matched at both viewport sizes and across cold, partial, and full CPU residency.

| Selected-value viewport | Cold worker, before → after | Partially resident worker, before → after | Fully resident worker, before → after |
|---|---:|---:|---:|
| Full extent: 60,512 points / 4,453 tiles | 97.15 → 98.48 ms | 73.60 → 77.46 ms | 40.46 → 42.15 ms |
| Centered 0.2 width/height fraction: 4,846 points / 247 tiles | 7.28 → 8.55 ms | 6.27 → 7.13 ms | 2.43 → 2.39 ms |

The all-values smoke run retained the same level-8, one-tile, 100,000-point snapshot. Cold worker time was 89.75 ms before and 86.26 ms in the first after-run; a subsequent final-code run measured 24.60 ms. Warm snapshots were 1.26 ms before and approximately 1.1–1.3 ms after. This variation illustrates the uncontrolled filesystem/codec-cache effects: these samples are correctness and cost checks, not evidence of a speedup or a statistical guarantee of zero regression. Selected-value trace process peak RSS was 327.47 → 333.50 MiB; retained compact NumPy metadata remains unchanged. No selection, value-major read, packing, or rendering algorithm was changed.

Diagnostic cost was measured separately on one 108,598-point Exact tile, with nine warm repetitions and allocation tracing disabled during timing. Returning one selected point took 12.02 ms for complete read plus filtering; returning 4,191 points took 11.81 ms. Filtering already-loaded arrays alone took 0.39–0.40 ms. Separate traced allocation peaks were approximately 3.14–3.20 MiB for complete read plus filtering and 0.42 MiB for filtering alone, including returned allocations rather than reporting process RSS. This cost is bounded to the requested complete tile and is outside normal proper-subset viewport routing.

Evidence: `/private/tmp/napari-harpy-slice10c-before.json`, `/private/tmp/napari-harpy-slice10c-after.json`, `/private/tmp/napari-harpy-slice10c-all-before.json`, `/private/tmp/napari-harpy-slice10c-all-after.json`, `/private/tmp/napari-harpy-slice10c-all-after-final.json`, `/private/tmp/napari-harpy-slice10c-bucket-smoke.json`, and `/private/tmp/napari-harpy-slice10c-diagnostic.json`. The diagnostic measurement driver is `/private/tmp/napari_harpy_slice10c_diagnostic.py`.

**Exit condition**

Complete-tile display reads have one descriptor-based addressing contract. No `_BucketLookupIndex`, sparse-subset display branch, or lookup-priming/accounting machinery remains. Diagnostics and equivalence references independently filter actual tile-major point arrays. The `_PointsCacheReader` docstring defines catalog metadata and documents reader ownership, both physical layouts, and when metadata versus payloads are read. Any test consolidation preserves the distinct addressing, validation, lifecycle, and IO guarantees. The current cache format, persisted sparse ranges, construction and validation guarantees, normal viewer routing, and physical batching remain intact. Optional Slice 17 is concerned only with the separate persisted-range removal decision and its remaining consumers.

### Slice 10d — Consolidate value-major array ownership in `_ValueMajorLevelReader`

**Status: Implemented**

This is a bounded reader-responsibility refactor following Slice 10c and preceding adaptive routing in Slice 11. It makes the value-major component easier to follow without changing the stored cache, physical read algorithm, or viewport behavior. Later slice numbers remain unchanged. It is not an expected loading/rendering-speed improvement or a cache-rebuild task.

This boundary also gives future point-aligned columns, such as quality scores, a clear home for their array references, layout checks, and physical reads. Adding those columns, changing payload contracts, or introducing a generic column framework is not part of this slice.

**Current responsibilities and intended boundary**

Before this slice, the work was split across three classes, not contained entirely in `_PointsCacheReader`:

| Responsibility | Before this slice | After this slice |
|---|---|---|
| Open and retain `value_major/level_N/location` and `value_major/level_N/value_point_indptr` Zarr objects | `_CacheRootReader` opens both into its generic array dictionary; `_ValueMajorLocationReader` borrows `location` | `_ValueMajorLevelReader` opens and retains both array references for its level |
| Validate the two arrays' complete storage layouts | `_CacheRootReader._validate_layouts()` | `_ValueMajorLevelReader`, using shared strict validation helpers |
| Load the compact point-pointer vector into a read-only NumPy array | `_PointsCacheReader._load_runtime_indexes()` via `_read_only_array()` | A level-reader operation supplies the vector; `_PointsCacheReader` still retains it for planning |
| Read bounded ordered location intervals | `_ValueMajorLocationReader.read_intervals()` | `_ValueMajorLevelReader`, preserving the existing implementation and cancellation boundaries |

Ownership here means responsibility for the Zarr array objects and their valid lifetime, not an additional copy of their contents. The loaded NumPy pointer vectors remain distinct from the stored arrays they describe.

**Implementation sequence**

1. **Expand and rename the existing location reader.** Replace `_ValueMajorLocationReader` with `_ValueMajorLevelReader`, rather than adding another wrapper around it. It owns the two Zarr array references for one level, validates their dtype, shape, chunk/shard layout, codec, and required array-attribute contract, supplies the compact pointer vector, and performs bounded location reads. Keep shared selection and validation helpers shared; do not introduce a second implementation or a compatibility alias for the old class.
2. **Delegate from the root reader while keeping one root store.** `_CacheRootReader` continues owning the root store/group, parsed `_CacheAttributes`, and the `manifest/*`, `values/n_points`, and `value_tiles/*` Zarr objects. It creates and retains one level reader per serialized level using that already-open root/group, exposes the existing level-reader instance to consumers, and delegates value-major layout validation to it. Remove the value-major entries from the root reader's generic array dictionary. Do not open an additional `LocalStore` per level or create separate level readers for validation and viewer access within the same root-reader lifetime. Preserve eager structural/layout rejection at root opening; location payloads remain unread until requested.
3. **Keep viewport semantics and NumPy residency in `_PointsCacheReader`.** Obtain the existing level readers through the root reader, load each compact `value_point_indptr` vector once at runtime startup, and retain/account for those read-only NumPy arrays exactly as before. Do not also retain a second pointer-vector copy in each level reader. LOD selection, `_SelectedValueIndex`, viewport-to-manifest intersection, resolving requested value-major intervals, reconstructing value IDs, scattering results into logical tiles, and CPU residency remain in their current layers. The new level reader accepts physical row intervals, not a viewport plan or selected-value index.
4. **Preserve independent validation and lifecycle guarantees.** Keep cache-wide reconciliation in `_CacheRootReader.validate_contents()`, obtaining pointer data through the level-reader boundary. In particular, retain validation of pointer origin, monotonicity, terminal count, and per-value differences against counts independently accumulated from `value_tiles`. Routine viewer startup must not begin running this full reconciliation or optional exhaustive validation. On normal closure and failed startup, release/invalidate the level readers' array references before closing the shared root store; partially initialized readers must not leak resources, and borrowed readers must not remain usable after their owner closes.
5. **Migrate consumers and explanations together.** Update imports, focused tests, worker test doubles, and benchmark hooks that refer to `_ValueMajorLocationReader` or obtain value-major arrays through `_CacheRootReader.array(...)`. Adapt `scripts/validate_multi_scale_cache_points_zarr_exhaustive.py` and related diagnostics to the new level-reader boundary without routing their independent tile-major reference through viewport assembly. Update `_PointsCacheReader` and storage-reader docstrings to distinguish root-store ownership, delegated level-array ownership, borrowed reader references, retained NumPy pointers, and returned location payloads. Retain explicit cache-relative paths. Keep `CACHE_FORMAT.md` focused on the unchanged persisted format.

**Boundaries and non-goals**

- No schema/version change, rebuild, compatibility path, in-place cache mutation, optional level, or new persisted array. Both value-major arrays remain mandatory at every level.
- No changes to `_BucketReaderCache`, complete-tile descriptor addressing, tile-major reads, sparse-range removal, physical routing, selection semantics, render budgets, packing, or GPU resources. Adaptive routing remains Slice 11 work.
- Preserve contiguous/basic and disjoint/orthogonal selections, bounded batch sizes, output order, cancellation checks, strict missing-chunk behavior, and the absence of eager location decoding. This refactor must not introduce extra payload reads, repeated pointer loads during pan/zoom, or extra stores.
- Root-level structural checks, value-major layout checks, compact runtime pointer checks, mandatory publication reconciliation, and optional exhaustive location-equivalence validation remain distinct guarantees. Moving ownership must not weaken them or increase the validation performed during viewer startup.

**Focused tests and verification**

- Adapt the existing location-reader tests to the level-reader contract and retain adjacent, disjoint, empty, multi-batch, invalid-interval, and cancellation coverage. Verify that opening the reader does not decode locations.
- Retain root-open rejection of missing/malformed level arrays, invalid dtypes/shapes/layouts/codecs, and unexpected attributes. Keep staged-validation tests that reject total or per-value pointer disagreement with `value_tiles`, and exhaustive-validation corruption cases with an independent tile-major reference.
- Verify reuse of the same per-level reader instances and shared root store, one runtime pointer load per level, unchanged resident-index byte accounting, no pointer reload during unchanged-selection viewport requests, and cleanup after normal closure or partial opening failure. Test these behaviors at the read/store boundaries, not only by asserting renamed class types.
- Run focused reader, value-major viewport-equivalence, validation, benchmark-helper, and cache-session tests affected by the new boundary. Preserve Exact, Bridge, and Spatial outputs, partial/missing-tile behavior, canonical value IDs, and normal-viewer no-sparse-read guards.
- Smoke-test migrated scripts/hooks. Compare startup and representative cold/warm worker requests on the same existing cache, checking output equality, physical read counts, compact resident bytes, and any additional allocation or latency. Treat this as a regression check, not evidence of a speedup; no cache construction or GPU benchmark is required solely for this ownership refactor.

**Implementation and verification (2026-09-08)**

- `_ValueMajorLevelReader` replaces the location-only reader without an alias. It opens and validates both level arrays through shared strict-array/layout helpers, supplies explicit read-only pointer loading, and preserves bounded interval reads and cancellation. It retains no decoded pointer vector or point payload.
- `_CacheRootReader` owns one shared store and registers each successfully constructed level reader immediately, so a later opening failure closes earlier readers. Its generic `array()` API now exposes only cache-wide lookup arrays. Normal and failed closure invalidate borrowed level readers before closing the store. `_PointsCacheReader` borrows these same instances and retains/accounts for each loaded pointer vector once.
- Publication pointer/count reconciliation remains separate from runtime startup. The exhaustive script reads its observed locations through the level-reader boundary while retaining its independent tile-major reference. Writer tests, viewport tests, benchmark hooks, and reader ownership documentation were migrated; the persisted format and `CACHE_FORMAT.md` schema description remain unchanged.
- **196 focused tests passed** across level/root/cache readers, lifecycle, viewport equivalence, bucket validation, catalog/value-major writers, staged/exhaustive validation, benchmark helpers, and worker sessions. Added coverage includes malformed array layouts/attributes, strict missing chunks, exact/bounded selectors, zero-payload-IO opening, one pointer load per level, unchanged residency through viewport changes, and normal/partial-failure cleanup. Ruff lint/format checks and `git diff --check` pass; dependency deprecation warnings remain.
- Existing-cache worker comparisons used five repeats at each of two viewport sizes, with empty, partial, and full CPU tile residency. **All 30 paired render-batch hashes, physical routes, and per-array IO counters matched.** Compact resident NumPy metadata remained **1,190,344 bytes**, including **368,856 bytes** of value-major pointers. Full-extent cold reads still selected 60,512 location rows in one operation, touching 16 chunks in one shard; fully resident requests performed no payload IO.

| Selected-value viewport | Cold worker, before → after | Partially resident worker, before → after | Fully resident worker, before → after |
|---|---:|---:|---:|
| Full extent: 60,512 points / 4,453 tiles | 98.03 → 91.97 ms | 72.39 → 70.86 ms | 40.13 → 41.74 ms |
| Centered 0.2 width/height fraction: 4,846 points / 247 tiles | 7.15 → 7.48 ms | 5.85 → 6.39 ms | 2.31 → 2.47 ms |

Single reader-startup samples were 347.41 → 155.68 ms; process peak RSS was 330.27 → 331.39 MiB. These are descriptive regression checks with uncontrolled filesystem/codec caching, not evidence of a startup speedup or a statistical guarantee of zero timing regression. Cold refers to CPU tile residency, not a flushed filesystem cache. No cache rebuild or GPU benchmark was performed.

Evidence: `/private/tmp/napari-harpy-slice10d-before.json` and `/private/tmp/napari-harpy-slice10d-after.json`, generated by `scripts/benchmark_tiled_points_viewport_planning.py`.

**Exit condition**

Each value-major level has one explicit reader responsible for its two Zarr array objects, layout checks, compact-pointer loading, and bounded location reads. `_CacheRootReader` owns the shared store and delegates value-major access instead of duplicating those array references in its generic dictionary. `_PointsCacheReader` remains the viewport coordinator and owner of its loaded runtime pointer vectors. Independent validation, resource cleanup, IO counts, memory-accounting semantics, and both viewer routes are preserved. No old reader-class alias or second location-reading implementation remains.

### Slice 11 — Measured adaptive proper-subset physical routing

**Status: Deferred — later loading-optimization phase**

Implement Slice 12 and establish the initial Slice 13 interaction baseline first, using the existing fixed routes. The design below is retained for later implementation, not a prerequisite for Slices 12–17. Revisit it against the measured remaining coverage misses; optional debounce, GPU hardening/scaling, and persisted-range removal need not all be implemented first. Completing the interaction milestone does not mark this slice implemented.

This is a follow-up optimization to the deliberately simple Slice 8 routing rule. It addresses the case where a proper subset contains enough values, and the viewport covers few enough tiles, that reading complete tile-major tiles plus point-level `value_id` and filtering in memory is physically cheaper than gathering many value-major intervals. It builds on Slice 9's lean semantic plan, Slice 10b's self-contained tile descriptors, Slice 10c's complete-only bucket display contract, and Slice 10d's per-level value-major reader boundary. It must not restore the sparse range indexes excluded from the viewer in Slice 10 and retired from diagnostic reads in Slice 10c.

The semantic selection and the physical payload route are separate decisions:

```text
all canonical values
        -> tile_major_all_values

proper subset
        -> value_major_subset
        or
        -> tile_major_filter
```

`tile_major_filter` means reading the complete row interval for each missing logical tile from the tile-major `location` and point-level `value_id` arrays, then retaining only rows whose value ID belongs to the requested selection. It does not resolve `ranges/{tile_indptr,value_id,row_start,row_count}` and does not reintroduce a sparse-index fallback cache. Both proper-subset routes must return the same ordered `_TileReadResult` contract, so CPU residency, render-batch packing, generations, cancellation, and VisPy remain independent of the chosen physical ordering.

“Proper subset” remains a semantic classification, not a cost heuristic. The complete canonical vocabulary is normalized to `requested_value_ids=None` and is therefore all-values; any normalized non-`None` selection containing fewer than `value_count` IDs is a proper subset. That classification determines which routes are eligible. The physical cost comparison then determines which eligible proper-subset route to use for this LOD, viewport, and set of CPU-residency misses.

**CPU residency is independent of the physical route**

Keep `TileResidencyKey` unchanged: cache generation, requested value IDs, level, and logical tile coordinates. It identifies a decoded, selection-specific `TiledPointsRenderTile`, not a Zarr chunk or a physical read route. Do not add route identity to this key, maintain separate CPU caches per route, or require acquisition history to reuse a tile.

Both readers must normalize their output before the viewer retains it. Value-major reads synthesize IDs and scatter locations into logical tile order; tile-major-filter reads discard unselected rows before returning the logical tile payload. Both therefore supply the same tile-relative locations and aligned value IDs for the same key. Extra complete-tile rows read for filtering are transient; they are not retained as an all-values tile under a selected-value key.

For example, with the same cache generation, selection, and level, tiles 10 and 20 may already be resident after a value-major read. A new viewport requesting tiles 10, 20, and 30 may choose tile-major-filter for missing tile 30. Reuse tiles 10 and 20 unchanged, then assemble all three payloads in logical order. This is compatible with one physical route per missing-tile request: the restriction applies to new reads, not to the acquisition history of resident tiles. The route may change on a later request without invalidating equivalent resident payloads.

A changed selection has a different residency key. This slice does not introduce cross-selection reuse or retain the unselected points from a complete-tile read for a future selection. Route diagnostics belong to the physical request, not the cache-hit correctness contract.

**Decision boundary**

Do not choose the route from `len(requested_value_ids)` or the selected-value fraction alone. Those values do not express viewport size, value distribution, compressed layout, or CPU-resident tiles. Final physical routing happens only after:

```text
semantic LOD selection
        -> positive logical tiles
        -> CPU-residency lookup
        -> missing logical tiles
        -> estimate both eligible physical reads
        -> choose one route for the complete missing-tile request
```

All-values requests remain unconditionally tile-major. A request with no missing tiles performs neither cost estimation nor physical reading. For a proper subset with missing tiles, compare estimates derived from the selected level's actual metadata:

| Candidate | Estimate from |
|---|---|
| `value_major_subset` | selected values' `value_point_indptr` intervals intersected with the missing manifest rows; unique location chunks or shards, decoded rows or bytes, disjoint runs, read operations, and blocks/rows to scatter |
| `tile_major_filter` | compact complete-tile intervals for the missing manifest rows; unique tile-major `location` and `value_id` chunks or shards, decoded rows or bytes, bucket operations, rows to filter, and bounded transient allocations |

The estimator must account for deduplicated physical chunks or shards rather than summing each logical interval independently. Count them separately per physical array and bucket; the same chunk number in different arrays or buckets is not shared work. Use schema metadata and integer arithmetic to derive work estimates; route selection must not perform speculative payload reads or expand intervals into one selector integer per point merely to estimate costs. Chunk count alone is insufficient because tile-major reads both `location` and `value_id`, while value-major reads only `location`. Read-operation estimates must follow actual batching, not assume one operation per interval or shard.

**Concrete estimation and dispatch**

1. Resolve the requested missing manifest rows in `read_planned_tiles()`. Return immediately if none remain; dispatch all-values requests directly to the existing complete tile-major reader without a cost comparison.
2. For a proper subset, describe the tile-major candidate from each `_TileDescriptor`'s `bucket_id`, `bucket_row_start`, and `n_points`. Those fields give complete point intervals without sparse-range lookup. Account for complete input payloads, row selectors, filtering masks, and filtered outputs when predicting temporary allocations; the selected-point budget alone is not a worker read bound. If this candidate exceeds the explicit bound, make it ineligible and use value-major without running a two-route comparison.
3. When both routes are eligible, resolve the value-major candidate from the selected level index, complete value/tile counts, and retained `value_point_indptr`. Extract the existing physical block resolution from `_read_value_major_requests()` into a shared helper. Resolve these blocks once for estimation and reuse the same resolved blocks if value-major is chosen; execution must not repeat their intersection and prefix-sum work.
4. Derive each candidate's physical-work and postprocessing estimates from its intervals and the stored layouts. Start with a small fixed cost model combining estimated read/decode work with route-specific filtering or scattering work. Determine the useful terms, coefficients, and units from paired forced-route benchmarks, not a selected-value threshold. The exact formula and coefficients are not yet fixed by this specification and must be recorded with calibration evidence before automatic routing is accepted. Resolution work paid by both candidates must not be charged as though it were unique to executing value-major; measure the total estimator overhead separately.
5. Execute the lower-cost eligible candidate, preferring `value_major_subset` on an exact tie. Introduce a switching margin only if benchmarks show that small estimate differences are noisy. The same immutable inputs and fixed estimator configuration must produce the same decision. Do not add timing history, per-tile acquisition provenance, adaptive state, or hardware-dependent feedback to the correctness contract. The score predicts relative work/latency; it is not a guarantee of actual wall time or a model of filesystem/codec cache residency.

Choose one route for the complete missing-tile batch initially. A per-tile or per-bucket hybrid could reduce physical work in a mixed case, but it would complicate ordering, cancellation, metrics, and testing; it requires separate evidence after this slice. The point budget still bounds returned points, not the number of complete tile-major rows decoded before filtering, so reject or avoid `tile_major_filter` when its predicted transient allocation exceeds the explicit worker read bound.

**Production changes**

1. Preserve `_ViewportReadPlan` as the generation-bound semantic plan. Refactor the fixed proper-subset route introduced by Slice 8 into a worker-local physical read plan resolved from the missing tile keys; do not move this decision to the coordinator, GUI, or renderer.
2. Add the `tile_major_filter` reader path using only compact complete-tile addressing, tile-major point arrays, and the immutable plan-wide `requested_value_ids` membership set. Read complete tile payloads through the same narrowed manifest reader used by the all-values route, then filter their point-level `value_id` rows in memory. Do not reconstruct `_PlannedTileRead.applicable_value_ids`, introduce another per-tile selected-value projection, or restore the sparse-range bucket path removed in Slice 10c. This route must never instantiate or load a bucket sparse lookup index.
3. Add deterministic cost-estimation helpers and reusable physical block resolution following the sequence above. Keep their units, assumptions, and estimation overhead explicit in diagnostics rather than hiding the decision behind a selected-value threshold. Do not duplicate block resolution between estimation and execution.
4. Preserve canonical output ordering, value IDs, and the existing route-independent CPU-residency contract. A request forced through either route must produce byte-equivalent logical tile keys, locations, and aligned `uint32` value IDs before CPU residency insertion. Filtering must finish before retention; no route-dependent cache keys or duplicate resident representations are introduced.
5. Apply cancellation checks while reading and filtering complete tiles, and retain the existing all-or-nothing publication behavior for a candidate snapshot.
6. Record the chosen route, both estimated costs, decoded-row and byte estimates, unique chunk or shard estimates, operation estimates, reason for an ineligible route, actual physical counters, and filter input/output row counts.
7. Keep an explicit force-route hook limited to tests and benchmarks so the two implementations and the automatic choice can be compared on identical requests. Do not expose it as a user-facing rendering preference.
8. Express estimation and dispatch in terms of the complete tuple of requested CPU-residency misses, not camera bounds or an assumed viewport-only plan. Slice 12 already supplies misses from a render coverage rather than only the exact visible viewport. Preserve that boundary and route-independent coverage/residency identities. Both active coverage hits and previous packed-batch hits bypass estimation, physical reads, and packing; only active-payload identity reuse guarantees no VBO replacement.

**Focused tests**

- All-values plans always use `tile_major_all_values`; they never enter the adaptive proper-subset comparison.
- A sparse value across a broad viewport selects `value_major_subset`, while a dense near-all-values selection in a small viewport can select `tile_major_filter` under controlled metadata.
- CPU-resident tiles are excluded before estimating either route, and an entirely resident request performs neither cost estimation nor physical reading.
- Verify mixed-acquisition reuse across successive requests with unchanged cache generation, selection, and level: retain tiles 10 and 20 from value-major, then request tiles 10, 20, and 30 while forcing tile-major-filter for missing tile 30. Assert that only tile 30 is read, the existing resident entries remain reusable, and the assembled logical payload matches the single-route reference. Also cover the reverse acquisition order. No physical-route field is needed in the residency key.
- Verify that tile-major-filter retains only selected rows under the selected-value key. A changed selection must not reuse that entry as though it contained all values read transiently from the tile-major arrays.
- Forced value-major and forced tile-major-filter reads return identical ordered logical payloads for Exact, Bridge, and representative Spatial levels.
- Forced tile-major-filter reads use only the plan-wide `requested_value_ids` membership set; patch any attempted per-tile selected-value reconstruction or non-`None` sparse-range bucket request to fail.
- The automatic route is deterministic at the crossover and exact-tie boundaries, including the documented tie preference or switching margin.
- Verify deduplicated interval-based work estimates, including shared chunks and distinct arrays/buckets, without speculative point reads or point-sized selector allocations. When value-major wins, verify that its resolved blocks are reused instead of performing physical block resolution a second time.
- Patch every sparse-range load and resolver to fail and prove that both physical routes still work.
- Predicted tile-major transient memory above the worker bound makes that route ineligible.
- Cancellation, stale generations, read failures, and empty filtered results preserve existing snapshot publication and rollback semantics.

**Benchmark and calibration evidence**

Force each eligible route, then run automatic routing for the same selections, viewports, and CPU-residency misses. Cover a sparse one-value request, several sparse values, a dense value, near-all-values subsets, small and full viewports, and Exact, Bridge, and representative Spatial levels. Report estimates beside actual physical calls, unique chunks and shards, decoded rows and bytes, filter/scatter work, estimation time, total worker wall time, transient memory, and chosen route. Include empty, partial, and full CPU residency so estimator overhead is assessed against the actual remaining read work.

Use the completed Slice 12 coverage policy and initial Slice 13 fixed-route report as the baseline. Add the forced-route comparisons, automatic decision metrics, and crossover measurements deferred from that report, then rerun its affected coverage-hit, coverage-miss, and real-interaction cases. Do not retune the estimator only against the earlier exact-camera-viewport requests or count an already established coverage-reuse improvement as an adaptive-routing gain.

Calibrate the estimator and any switching margin from these paired measurements. Record the adopted formula, coefficients, units, and limitations; the existing equivalent-output tests establish the payload contract, not the performance crossover. Acceptance requires that automatic routing, including its estimation overhead, avoids clear regressions around the crossover and improves at least one demonstrated dense-subset case. A fixed number-of-values threshold is not acceptable evidence because the same selection can favour different layouts at different viewports or LODs.

**Exit condition**

Every all-values request remains tile-major. Every proper-subset physical read is selected once, after LOD and CPU-residency lookup, between the mandatory value-major sidecar and complete tile-major reads plus in-memory filtering. The choice is deterministic, measurable, bounded, produces the same logical payload, and never loads a sparse bucket range index. Existing CPU entries remain reusable regardless of which route originally supplied them; route changes do not invalidate residency or retain extra unselected points.

### Slice 12 — Stable render coverage and bounded packed-batch reuse

**Scheduling: Next implementation slice; independent of deferred Slice 11.**

This required interaction slice removes the performance cliff created by treating the exact camera viewport as the render-payload lifetime. It generalizes whole-selection reuse rather than adding a special fast path that works only below the 100,000-point boundary.

Keep the current physical routing unchanged: all-values coverage misses use complete tile-major reads, and proper-subset coverage misses use the selected level's value-major cache. `read_planned_tiles()` already accepts explicit missing logical tile keys, so expanded coverage does not require an adaptive route estimator. Coverage identity and CPU-residency identity must remain independent of the physical route, allowing deferred Slice 11 to optimize those misses later without redesigning coverage reuse.

**Measured motivation**

The value-major sidecar makes cold payload loading acceptably fast, but CPU residency alone does not make a warm viewport request cheap. After the complete AAMP Exact payload was resident, a full → quarter-width → full trace measured:

| Request | Rendered points | Logical tiles | Zarr payload reads | Worker snapshot |
|---|---:|---:|---:|---:|
| Initial full extent | 60,512 | 4,453 | 1 | 114.7 ms |
| Zoom in | 8,213 | 392 | 0 | 5.1 ms |
| Zoom back out | 60,512 | 4,453 | 0 | 56.4 ms |
| Repeated zoom back out | 60,512 | 4,453 | 0 | 53.7 ms |

The repeated warm full request spent approximately 17.8 milliseconds in viewport-plan construction, 20.3 milliseconds packing the complete vertex batch, 2.4 milliseconds in 4,453 CPU-residency lookups, 1.1 milliseconds validating ordered render tiles, and the remaining approximately 12 milliseconds constructing keys, tuples, dictionaries, and snapshot state. The physical value-major array was not accessed.

A boundary case demonstrates why a binary whole-selection optimization is insufficient:

```text
AAMP   60,512 Exact points
CRYZ   39,594 Exact points
total 100,106 Exact points
```

Exceeding the hard budget by only 106 points makes the current first-fit LOD policy select Bridge. That Bridge request contains only 12,755 selected points, but they remain fragmented across 7,112 value/tile records and 4,820 positive logical tiles. Its initial full snapshot measured 138.4 milliseconds and a fully CPU-resident zoom back out still measured 60.5 milliseconds with zero Zarr reads: approximately 20.5 milliseconds of planning, 22.6 milliseconds of packing, and 17 milliseconds of remaining tile-level work. A point-count budget therefore bounds vertex rows but does not bound preparation work or visual latency.

The visual symptom follows directly from viewport-scoped coverage. During zoom out, the camera immediately exposes space beyond the active smaller snapshot. A later complete snapshot installs a larger tile-aligned payload, and a sequence of accepted intermediate snapshots can look like tiles arriving even though the renderer uses one atomic VBO replacement per snapshot.

**Target interaction model**

Keep the camera viewport, semantic LOD, render coverage, CPU residency, and physical storage route as separate decisions:

```text
camera viewport
        |
        v
semantic LOD from visible points
        |
        v
active coverage still valid? ---- yes ----> reuse active payload
        |                                      acknowledge latest generations
        no                                     no reads, pack, or VBO replacement
        v
previous cached coverage valid? - yes ----> reuse its immutable CPU batch
        |                                      acknowledge latest generations
        no                                     no reads or pack; upload if needed
        v
choose deterministic tile-aligned coverage within hard budgets
        |
        v
CPU-residency lookup for coverage tiles
        |
        v
existing physical reader for coverage misses only
        |
        v
pack and retain one immutable coverage batch
        |
        v
replace the existing single VBO once
```

A render coverage is a tile-aligned spatial region at one selected LOD containing every requested value row in its included logical tiles. It is normally larger than the camera viewport. The GPU may process points outside the visible rectangle, but the normal scene transform and clipping prevent them from appearing. The complete coverage payload, not merely the current camera bounds, is the unit of physical identity and reuse.

Distinguish three outcomes after the visible LOD/count check:

- **Active coverage hit:** the current GPU payload already covers the visible tile footprint with the required cache generation, selection, LOD, and valid budgets. Reuse it without full tile-plan construction, per-tile CPU-residency lookup, Zarr payload reads, packing, or VBO replacement. Camera transforms and drawing still occur; the newest logical generations and status are acknowledged.
- **Previous packed-batch hit:** the immediately previous retained CPU coverage is valid for this request. Reuse its packed allocation without rebuilding its tile plan, fetching decoded tiles, reading payloads, or packing. If its physical identity differs from the active GPU payload, upload it into the same VBO before activation. A CPU batch-cache hit does not imply a GPU-payload hit.
- **New coverage construction:** neither retained coverage can satisfy the request. Choose a bounded coverage, construct its tile plan, reuse decoded CPU-resident tiles, read only missing ones, pack once, and stage the resulting payload in the single VBO. Having all decoded tiles resident still avoids only the physical reads in this case, not coverage planning and packing.

Use these outcome names in tests and metrics. In particular, the no-VBO-replacement guarantee applies to an **active coverage hit**, not to every packed-batch cache hit. Both reuse cases retain the same selection/LOD validity and stale-generation checks; neither can reactivate an obsolete result merely because its allocation is cached.

Whole-selection rendering becomes the natural maximum-coverage case:

- AAMP alone has 60,512 Exact points, so its Exact coverage can span the complete dataset and ordinary pan or zoom events require no payload change.
- AAMP plus CRYZ has 100,106 Exact points, so complete Exact coverage is ineligible. At full extent the semantically selected Bridge representation contains only 12,755 points and can itself use complete-dataset coverage. A zoomed-in Exact view receives a bounded expanded Exact coverage around the camera instead of reverting to an exact-viewport-only payload.

This produces one policy on both sides of 100,000 points rather than a fast branch below the limit and the current stuttering path above it.

**Snapshot counts and validation**

Update `TiledPointsRenderSnapshot`, its validation, documentation, and consumers to distinguish the latest visible-viewport estimate from the complete coverage payload count:

- The **visible-point estimate** drives the camera's LOD/budget decision and viewport status. Preserve the existing meaning: selected points in complete logical tiles intersecting the viewport, not a new point-by-point clipping count.
- The **coverage-point count** describes every selected row in the packed coverage, including off-screen tiles. It must equal `render_batch.point_count`, supply the expected `point_count` when packing a new coverage, and govern the complete payload's hard point/vertex-byte checks. `rendered_tile_count` must describe that same coverage tile set.

The current within-budget validation requires `render_batch.point_count == estimated_point_count` because one viewport estimate also describes the complete packed payload. Do not retain that equality against a newly defined visible-only estimate. Introduce unambiguous separate meanings in the contract, preserve count/dtype/byte validation for the full packed payload, and update status and budget consumers accordingly. A visible estimate of 20,000 with a 60,000-point coverage is valid if both applicable budgets are satisfied; validating only the 20,000 visible estimate would not protect the renderer from an oversized coverage.

On either reuse path, refresh the visible estimate, omission/status information, and request/selection generations while retaining the cached coverage counts and allocation unchanged. Metadata-only over-budget results must still carry an empty batch rather than pretending an invalid visible request is a successful coverage activation. Tests must exercise differing visible and coverage counts rather than relying only on full-extent requests where they coincide.

**Production changes**

1. Preserve the current 100,000-point hard render limit and `max_vertex_payload_bytes`. Coverage may include off-screen points only while its complete point and vertex-byte totals remain within both limits. Do not raise either limit or hide over-budget rows in the renderer.
2. Continue choosing semantic LOD from the visible camera viewport and selected values. Off-screen coverage rows must not inflate the visible estimate and force a coarser LOD. Once a level is chosen, coverage expansion occurs only within that level.
3. Add a worker-owned immutable render-coverage contract containing at least cache generation, requested value IDs, selected LOD, deterministic tile-aligned coverage bounds, ordered logical tile identity, coverage-point count, and packed-batch identity. Update the snapshot count contract and consumers as specified above; the latest visible-point estimate is distinct from the packed coverage count. Request and selection generation counters, continuous camera bounds, visible estimates, status text, and omission metadata are logical activation state rather than physical identity.
4. On every camera request, perform the inexpensive visible LOD/count check first. If cache generation, selection, and LOD still match and the complete visible tile footprint lies inside active coverage, reuse the existing immutable packed batch. Return or publish the newer logical generations and status as a successful result without calling the complete viewport planner, `pack_render_tiles()`, or `VertexBuffer.set_data()`.
5. When the chosen level's complete selected payload fits both hard budgets, use complete-dataset coverage. This rule applies to Exact, Bridge, and every Spatial level; it is not an Exact-only or one-gene optimization.
6. Otherwise, start with the complete visible tile envelope and grow deterministic complete tile rings around it while the next complete ring fits both budgets. Stop at the last complete rectangle rather than selecting a nondeterministic collection of spare tiles. Record unused headroom and why growth stopped. The visible envelope itself must already satisfy the semantic LOD budget before coverage expansion begins.
7. Add calibrated LOD hysteresis. Coarsen whenever the active finer level would exceed the hard limit; never violate the limit. Refine again only when the finer visible estimate lies below a measured lower watermark that leaves useful coverage headroom. Initial requests and selection changes still choose the finest valid level. Do not choose watermark values from intuition; calibrate them with the boundary traces in this slice and record their quality/latency trade-off.
8. Define one canonical GUI-neutral physical-payload identity from cache generation, requested value IDs, selected LOD, and ordered coverage tile keys. Include cache, selection, and LOD even for an empty tile tuple. Do not hash or scan vertex rows on the GUI thread.
9. Treat an active-identity match as a successful activation. The latest request and selection generations, omission information, and status must be acknowledged even though packing and VBO replacement are skipped. A stale or failed candidate cannot commit either logical state or reusable identity.
10. Retain a worker-owned byte-bounded cache of at most the current and immediately previous immutable packed coverage batches. After an active coverage miss, check whether the previous cached coverage is valid before constructing a new coverage tile plan or performing per-tile CPU-residency lookup. This specifically supports common full → partial → full and adjacent-LOD reversals without retaining an unbounded viewport history. Key entries by the canonical physical identity, account their bytes separately from decoded CPU tile residency, clear unrelated entries on cache or value-selection changes, and use deterministic MRU eviction when both legal batches do not fit the configured bound.
11. Preserve queued-allocation identity when a cached batch is delivered again. The batch remains read-only for its complete worker, Qt-delivery, and renderer lifetime. The renderer retains exactly one visual, one program, one VBO, and one point draw; returning to a cached CPU batch may upload it into that VBO, but it must not allocate another GPU-resident tile set or VBO cache.
12. For new coverage construction, establish coverage before CPU-residency lookup, then pass only nonresident coverage tile keys to the existing `read_planned_tiles()` boundary. Preserve its fixed all-values/tile-major and proper-subset/value-major routes; do not add a cost estimator, production filtering route, or placeholder adaptive-routing layer here. Active coverage hits and previous packed-batch hits bypass physical readers entirely, but only active-payload identity reuse guarantees no VBO replacement. Deferred Slice 11 can later choose a route for this same complete miss set without changing coverage or CPU-residency keys.
13. Keep current decoded CPU tile residency useful for constructing a new coverage, but do not mistake thousands of tile hits for a cheap warm request. Instrument plan construction, key creation, batch lookup, validation, and packing by both point and tile count. If uncached coverage construction remains a material interaction stall, replace per-tile dictionaries and small NumPy arrays with immutable aligned plan arrays and use a measured whole-batch or blocked packing strategy; do not solve preparation overhead by silently selecting a coarser LOD or imposing an unsupported hard tile-count limit.
14. Keep coverage planning, cache lookup, and packing on the worker. Because Python-heavy work on a `QThread` can still contend with the GUI for the GIL, record main-thread frame gaps while constructing an uncached highly fragmented coverage. Preserve cooperative close cancellation and latest-generation rejection throughout coverage construction and packed-cache reuse.
15. Do not add viewport debounce in this slice. First remove repeated accepted work through coverage and identity reuse. Slice 14 may add a short debounce only if the integrated camera trace still proves material obsolete dispatch churn.

**Focused tests**

- A complete selected Exact payload below the hard limit becomes dataset-wide coverage; repeated pan and zoom requests reuse the same packed allocation and never call physical readers, the complete viewport planner, the packer, or VBO replacement.
- A selection just over the Exact limit uses a legal coarser full-dataset coverage when that selected level fits. It must not fall back to the under-budget special case or exceed the hard limit.
- A zoomed-in Exact request whose complete selection exceeds the limit builds deterministic expanded coverage around the viewport, never exceeds either budget, and reuses it while camera bounds remain inside.
- Crossing the active coverage boundary first checks the previous retained coverage. A valid previous packed-batch hit reuses that allocation; otherwise construct one new coverage. Moving within valid active coverage does neither. Exact boundary equality, empty intersections, dataset edges, highly uneven tile counts, and a ring that would exceed the remaining budget are deterministic.
- LOD hysteresis never allows more than 100,000 visible or coverage points, prevents repeated Exact/Bridge oscillation around the boundary, and refines again at the documented measured lower watermark.
- Same physical identity with newer request/status metadata is acknowledged without packing or VBO replacement. Changed cache generation, value IDs, LOD, tile membership, or tile order invalidates identity.
- Visible estimates smaller than coverage counts pass snapshot validation when both are valid. Movement inside the same coverage updates visible estimates and logical metadata without changing packed counts or allocations. Reject a mismatched coverage/batch count and an over-budget coverage even when the visible estimate fits; retain empty-payload validation for metadata-only over-budget results.
- Current → previous → current coverage reuses the two cached immutable allocations without full tile-plan construction, per-tile CPU lookup, payload reads, or packing. Switching to a different cached identity still stages it into the single VBO; an active identity match does not. A third distinct coverage, selection change, cache close, byte-bound failure, and empty payload exercise deterministic eviction and cleanup.
- Cached allocation identity survives queued Qt delivery. All retained batches are read-only and their complete byte footprint is reported separately from CPU tile residency and the single GPU VBO.
- Missing decoded tiles during new coverage construction use the current complete tile-major route for all values and the current value-major route for proper subsets. Compare selected logical payloads with the existing independent batched tile-major-plus-filter reference, without adding a production `tile_major_filter` route or requiring a force-route hook. Active coverage hits and previous packed-batch hits call neither physical reader and never load a sparse bucket index. Cross-route coverage tests using the production adaptive dispatcher belong to deferred Slice 11.
- Stale generations, cancellation, selection changes during construction, packing failure, VBO failure, and close cannot commit a coverage or packed-cache entry incorrectly.
- The renderer remains one visual, one VBO, and one draw submission across active coverage hits, previous packed-batch hits, new coverage construction, and LOD transitions.

**Benchmark and calibration evidence**

Record real camera traces and deterministic full → partial → full replays for:

- AAMP alone at 60,512 Exact points;
- AAMP plus SEC16A at 99,998 Exact points;
- AAMP plus CRYZ at 100,106 Exact points;
- selections comfortably above the boundary;
- Exact/Bridge transitions and representative Spatial transitions; and
- both sparse and dense spatial distributions with similar returned point counts but different tile counts.

For each request report camera events, LOD estimates, hysteresis decisions, visible and coverage bounds, visible and coverage tile/point counts, coverage headroom, the distinct outcome (active coverage hit, previous packed-batch hit, or new coverage construction), CPU-residency misses, physical route and reads, planning and packing time, avoided packed bytes, VBO replacements and bytes avoided, GUI frame gaps, Qt delivery, warm draw, and total interaction latency. Report uploads on previous packed-batch hits separately from the no-upload active-hit path.

Acceptance requires:

- no binary interaction-latency cliff immediately below versus immediately above 100,000 Exact points;
- active coverage hits to perform no Zarr read, per-tile CPU lookup, tile-proportional full-plan construction, snapshot packing, or VBO replacement;
- previous packed-batch hits to avoid those reads, lookups, full-plan construction, and packing while permitting an upload when the active GPU identity differs;
- the AAMP full selection to remain one stable Exact payload during ordinary camera movement;
- the slightly over-budget AAMP-plus-CRYZ trace to avoid repeatedly rebuilding its 4,820-tile Bridge payload;
- hard point and vertex-byte limits to remain exact;
- repeated full → partial → full and boundary-oscillation traces to lose the visible tile-arrival effect after their bounded coverages are prepared; and
- no regression in selection correctness, LOD omission reporting, transforms, generation handling, memory accounting, or the constant GPU-resource topology.

The first construction of a genuinely new coverage may still perform bounded asynchronous IO and packing. Its cost must be reported separately from active coverage hits and previous packed-batch hits. If the uncached transition still causes unacceptable GUI frame gaps, the tile-plan and packing representation in production change 13 must be optimized before this slice is accepted; debounce must not be used to conceal an expensive accepted request.

**Exit condition**

The exact camera viewport no longer defines the lifetime of the renderer payload. Every accepted view either reuses a valid budget-bounded coverage or constructs one deterministic replacement through the existing fixed physical routes. Whole-selection reuse works at any selected LOD that fits, selections just above 100,000 points do not fall back to a permanently stuttering path, LOD transitions are hysteretic and budget-safe, at most two explicitly byte-bounded immutable CPU batches are retained, and the renderer still owns one visual and one VBO. Adaptive routing is not required to accept this slice; remaining cold-read limitations are measured separately in Slice 13.

### Slice 13 — Integrated all-level acceptance and tuning matrix

This slice consolidates evidence for the mandatory all-level dual ordering and tunes its physical layout without making individual level sidecars optional.

The initial checkpoint follows Slice 12 without waiting for Slice 11: production routing remains all-values/tile-major and proper-subset/value-major. Establish an honest interaction baseline that includes remaining cold and dense-subset costs. Route-cost estimates, production forced-route comparisons, filtering metrics, and the adaptive crossover are deferred to Slice 11, which later extends this report and reruns the affected matrix rather than blocking its initial completion.

**Benchmark matrix**

Run the same cache generation and renderer across:

- sparse one-value AAMP, full extent and partial viewport;
- at least one dense value;
- several selected values and near-all-values proper subsets;
- all values;
- Exact, Bridge, and representative spatial LOD decisions;
- cold application caches and repeated warm CPU-resident requests;
- full → partial → full camera transitions;
- the 99,998-point and 100,106-point Exact-selection boundary cases from Slice 12;
- movement wholly inside an active coverage, movement across one coverage boundary, and return to the immediately previous coverage;
- repeated motion around an Exact/Bridge and representative Spatial LOD boundary;
- 100,000-point real-canvas rendering; and
- synthetic 1,000,000-point packing only, without raising the product budget.

For every case report:

```text
LOD and physical route
visible and coverage bounds, tile counts and point counts
coverage budget headroom and expansion stop reason
active coverage hits, previous packed-batch hits, and new coverage constructions
LOD hysteresis decisions and refinement watermark
current/previous packed-batch cache hits and retained bytes
planned/returned tile and point counts on coverage misses
physical calls, chunks, shards, decoded rows and bytes
coverage-miss read/assembly peak transient bytes
sparse-index load count and resident bytes, both expected to remain zero
CPU residency lookup/read/retain time
viewport events, dispatched requests and accepted snapshots
accepted snapshots whose coverage identity matches the active payload
worker pack time and peak transient bytes
Qt delivery time and allocation identity
VBO staging time and active bytes
VBO replacements and upload bytes avoided
visual/VBO/draw count
first and warm physical draw
main-thread frame gaps and end-to-end interaction latency
process RSS at startup, snapshot, staging and first draw
```

**Decision rules**

- Treat every serialized level's mandatory sidecar as part of the accepted cache format; use the matrix to tune per-level physical layout and quantify cost rather than deciding whether newly built caches may omit individual levels.
- Require proper-subset production reads to use value-major and all-values reads to use complete tile-major. Reject implicit fallback caused by a slow or corrupt sidecar and any sparse-range loading; fix the sidecar or reject the cache generation. Independent diagnostic/reference tile-major filtering remains valid, but is not a production adaptive route.
- Validate that the existing physical readers receive only nonresident tiles during new coverage construction, while active coverage hits and previous packed-batch hits invoke neither reader. Keep their VBO behavior distinct: reactivating a different previous CPU batch may require an upload. Preserve independent logical-payload equivalence checks. Defer forced value-major versus production tile-major-filter comparisons, automatic-choice metrics, and crossover acceptance to Slice 11; do not introduce that implementation solely to satisfy this checkpoint.
- Record cold coverage-construction and dense-subset limitations even when warm coverage reuse meets its targets. These costs inform whether and when to revisit Slice 11; a coverage-hit speedup is not evidence that the value-major route is always cheapest.
- Accept the coverage policy only if it removes the interaction-latency cliff around the 100,000-point boundary, stays within both hard budgets, and makes repeated motion inside an active coverage independent of its logical tile count. Calibrate the LOD-refinement watermark and packed-batch byte bound from the recorded boundary and reversal traces rather than from isolated microbenchmarks.
- Do not substitute smaller chunks, fewer buckets, or cross-bucket threading for the sidecar unless new end-to-end evidence contradicts the existing results.
- Do not increase the point budget based on packing time alone.

**Exit condition**

Publish one comparison report containing the pre-change baseline and each accepted slice. Record per-level construction size, read amplification, latency, sidecar tuning decisions, remaining fixed-route limitations, and the measured coverage/hysteresis behavior immediately below and above the 100,000-point boundary. This initial checkpoint can complete while Slice 11 remains deferred; its later acceptance adds the measured proper-subset routing crossover and checks for regressions against this interaction baseline.

### Slice 14 — Conditional viewport debounce

Debounce is deliberately late because it avoids work but does not make an accepted request cheaper.

Evaluate against the initial Slice 13 fixed-route baseline; Slice 11 is not a prerequisite. If the entry condition is not met, record the decision not to implement debounce. Neither that implementation nor the decision to defer it blocks the interaction milestone or later adaptive-routing work.

**Entry condition**

Proceed only if Slice 13 instrumentation shows that rapid camera gestures still dispatch multiple coverage constructions or physical reads that become obsolete despite stable coverage reuse and the existing one-active/one-latest-pending mailbox.

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

### Slice 15 — Optional hardening and scaling gates

These are explicit decision gates, not assumed follow-up work.

Evaluate them from the initial Slice 13 interaction measurements without requiring Slice 11. The current milestone retains one VBO and the 100,000-point hard budget unless a separate gate is justified and accepted. Neither a second VBO nor a larger product budget is required before revisiting adaptive routing.

#### Optional second VBO

Add a second VBO to the existing single snapshot visual/program only if repeated real-canvas replacements show active-buffer synchronization stalls, blank/corrupted frames, or a product requirement for stronger post-mutation recovery. Evidence must show that ping-pong storage fixes the observed issue. Keep one visual and one draw submission.

#### Optional 1,000,000-point product budget

Do not raise the budget until end-to-end tests cover worker packing, Qt delivery, VBO staging/deferred upload, repeated changing payload sizes, first and warm physical draws, fragment overdraw at representative point diameters, peak RSS, cancellation latency, and visual correctness. The 11.44-MiB logical vertex size alone is not sufficient evidence.

#### Deferred cache simplifications

Persisted bucket sparse-range removal, including the smaller `ranges/row_start`-only alternative, is evaluated separately in optional Slice 17. Quantizing coordinates, adding lazy per-value sidecars, using an uncompressed memory-mapped payload, or offering a value-major-only cache profile each changes another contract. Evaluate them only after the mandatory dual-ordering implementation has measured results, and keep each in its own schema/benchmark slice.

### Slice 16 — Explicit coordinator selection arming

This slice is a lifecycle and API cleanup rather than a rendering optimization. It makes the product rule explicit: selecting or inspecting a points element may load metadata and available values, but a regular or tiled napari points layer is created only after an explicit Add/Update action.

This work is independent of deferred Slice 11 and conditional Slices 14, 15, and 17. It can follow the initial Slice 13 baseline; after implementation, rerun the affected first-selection, generation, and coverage-reuse checks. It is not expected to reduce ordinary warm pan/zoom latency.

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

### Slice 17 — Optional removal of persisted bucket sparse ranges

This is a conditional cache-schema cleanup, not another viewer-residency or rendering fix. Slice 10 removes sparse lookup allocation from normal viewer startup and reads; Slice 10c removes the resident lookup object and diagnostic sparse-subset APIs entirely, without deleting stored ranges. Deferred Slice 11's proposed `tile_major_filter` also uses complete-tile addressing plus point-level filtering and does not require persisted sparse ranges. The potential benefits here are a smaller published cache and simpler persisted-schema handling; do not claim the earlier 568.4-MiB resident allocation as an additional saving or as the compressed disk space recoverable by this slice.

**Entry condition and scope decision**

Evaluate after Slice 13 has established the initial integrated viewer baseline, without requiring Slice 11. This is lower priority for stutter and is not a prerequisite for either the interaction milestone or adaptive routing. First measure the compressed size of each bucket `ranges` array and audit its remaining consumers. Proceed only if the storage and maintenance benefits justify the construction and validation changes. Retaining the current published schema is an acceptable outcome if removal merely shifts cost or adds complexity.

The full-removal target is bucket `ranges/{tile_indptr,value_id,row_start,row_count}`. Keep compact complete-tile addressing, the manifest, `value_tiles`, per-level value-major pointers, and both physical point payloads. In particular, neither point-level tile-major `value_id` nor the active selected-value `value_tiles` index is part of this removal.

If full removal is not justified, evaluate `ranges/row_start` alone as a smaller, separately scoped alternative: reconstruct each start from the tile offset and preceding `ranges/row_count` values within that tile. That alternative retains the other sparse-range arrays and must not be reported as complete sparse-range removal.

**Construction and reader changes if full removal is justified**

1. Retain the information needed to build the cache without retaining it permanently in every published bucket. `_iter_bucket_range_batches()` currently supplies catalog generation with `(value_id, manifest_index, row_start, n_points)` records; `write_value_tiles_by_level()` sorts them and preserves aligned source addresses in `ordered_row_start` for value-major construction. Replace that persisted-range dependency with bounded construction-only records or streams. Do not eliminate the source addresses before the value-major writer has consumed them.
2. Keep construction records disk-backed or streamed as needed, under build-owned temporary storage. Do not replace the persisted arrays with an unbounded all-level in-memory allocation. Retain temporary records until their construction and validation consumers finish, then clean them up on success and failure; they must not leak into the completed cache.
3. Audit diagnostics, benchmarks, and reference tests for any remaining direct dependency on persisted ranges. Slice 10c already replaced sparse-subset display reads with complete tile-major reads and independent in-memory filtering; preserve that reference independence. Do not redirect an equivalence test's reference to the same value-major implementation it is supposed to verify.
4. Remove persisted-range-specific reader/layout handling and tests only after their construction and validation consumers have replacements. `_BucketLookupIndex`, its loading/accounting machinery, and sparse-subset display resolution have already been removed in Slice 10c and are not additional work or savings here. Preserve compact complete-tile reads, coordinated bucket batching, canonical output order, cancellation, and both viewer routes.
5. Update bucket writers, schema constants, attributes and count/layout checks, strict hierarchy validation, and `CACHE_FORMAT.md` together. This is a new published schema requiring rebuilt caches, not an operation that deletes arrays from an existing user cache in place. Do not add deprecated aliases or silently accept a partially converted layout.

**Validation contract must be decided explicitly**

The current `_validate_bucket_ranges_against_catalog()` independently reads bucket ranges and compares their value/tile records with `value_tiles`. `_iter_compact_bucket_range_batches()` also supplies source addresses to the optional exhaustive location-equivalence validator. Neither consumer can simply be removed without reviewing what its checks prove.

- Construction-time reconciliation may consume the temporary range records before they are discarded. Document which checks then apply only during a build.
- Define bounded routine validation for a completed cache without relying on temporary files. Preserve the retained schema, count, pointer, and layout checks, and explicitly document any range-to-catalog comparison that is replaced or no longer available. Do not silently weaken validation or introduce an expensive mandatory point-array scan merely to compensate for removed metadata.
- Adapt optional exhaustive validation to recover actual value runs and bucket row addresses from tile-major point-level `value_id` and complete-tile boundaries in bounded batches, including runs crossing batch/chunk boundaries. It must independently verify the catalog and value-major locations; deriving both expected and actual records solely from `value_tiles` would not replace the lost cross-check. Exhaustive payload scans remain opt-in, not viewer-startup or routine-publication requirements.

**Focused tests**

- Full-removal builds publish no bucket `ranges` group and leave no construction-only records in the cache; temporary cleanup also succeeds after injected construction/validation failures.
- Catalog totals, value-to-tile records, and value-major locations remain correct at Exact, Bridge, and Spatial levels, including multiple buckets, repeated values, and runs spanning bounded batches.
- Complete tile-major reads remain correct, and selected value-major reads match the existing independent batched tile-major-plus-filter reference for complete, partial, and missing-tile requests without sparse-index loading. Do not require a production `tile_major_filter` route or force-route hook while Slice 11 is deferred. If adaptive routing has been implemented by this point, additionally run its forced-route equivalence checks against the new schema.
- Diagnostic/reference results remain independently derived from tile-major point arrays rather than the route under test.
- Routine validation rejects malformed retained metadata without requiring full point-payload reads. Optional exhaustive validation detects incorrect value counts, ordering, and locations after reopening a completed cache with no temporary records available.
- If only `row_start` is removed, test its reconstruction and validation separately, with correct offsets across tile, bucket, and level boundaries.

**Benchmark evidence and exit condition**

Compare compressed cache bytes by array/group, construction wall time, peak RSS, temporary-disk peak, routine validation time, and optional exhaustive-validation time. Verify unchanged viewer payloads, startup behavior, and representative cold/warm reads against the accepted runtime baseline. Measure the retained diagnostic/reference path separately, since complete-tile filtering can read more point rows than its former sparse-range path.

Accept full removal only when all published-range consumers have replacements, the new validation guarantees are explicit, no normal viewer path regains sparse lookup residency, and the measured storage/maintenance benefit justifies the cost. Otherwise retain the ranges or pursue the explicitly narrower `row_start` alternative. This optional decision is not a prerequisite for completing the visualization plan.

### Definition of done — interaction milestone and deferred routing

The current interaction milestone, including explicit selection arming, is complete when:

1. the default in-memory backend remains unchanged;
2. opt-in tiled rendering uses one visual, one VBO, one worker-prepared batch, and one draw submission;
3. GUI activation contains no tile-proportional packing or resource loop;
4. CPU residency no longer has quadratic no-eviction behavior;
5. after Slice 10c, bucket display batches accept only complete-tile requests with validated descriptor addressing; selected-value filtering occurs above that boundary, and there is no alternate sparse-lookup initialization or display mode;
6. value-major proper-subset reads never decode point-level `value_id`; complete tile-major all-values reads and independent diagnostic/reference filtering remain separate from that production subset path;
7. every newly built current-schema cache contains a structurally and index-validated value-major location sidecar for every serialized level, and that sidecar remains an eligible proper-subset route after LOD selection;
8. all-values coverage misses retain tile-major routing and proper-subset coverage misses retain value-major routing; no adaptive route estimator is required for this milestone;
9. no viewer startup or read path projects, loads, or retains bucket sparse-range indexes;
10. after Slice 10c, the resident bucket lookup and diagnostic sparse-subset reader are removed, while persisted bucket sparse ranges remain available for cache construction, catalog generation, and independent validation; diagnostic/reference subset reads independently filter complete tile-major point arrays. Optional Slice 17 may change the persisted contract only after adapting its remaining consumers and validation, and neither choice permits a viewer sparse-index fallback;
11. every accepted camera view is contained by a deterministic, budget-bounded render coverage; active coverage hits avoid physical reads, tile-proportional planning, packing, and VBO replacement. Previous packed-batch hits avoid the reads, full-plan construction, and packing but may upload into the single VBO. Visible estimates remain distinct from full coverage counts, and LOD hysteresis prevents repeated boundary oscillation without ever exceeding the hard limits;
12. benchmark reports demonstrate improved cold reads, warm activation, coverage-hit interaction, first draw, warm draw, startup RSS, steady memory, and no latency cliff immediately below versus above the 100,000-point boundary;
13. the tiled coordinator distinguishes selection-not-configured from an explicit all-values selection, and its first cache read is armed only by the explicit Add/Update path; and
14. debounce, ping-pong storage, alternative sidecar encodings, a larger point budget, and optional persisted sparse-range removal are accepted only when their own evidence gates are met.

The initial Slice 13 baseline can be published before the independent Slice 16 cleanup; finish its lifecycle acceptance checks before declaring the milestone above complete. Conditional Slices 14, 15, and 17 may be declined or deferred without blocking completion. Their eventual implementations require their own regression checks and do not redefine the fixed-route baseline retroactively.

Deferred Slice 11 remains separate, unfinished work after this milestone. Its later completion requires one deterministic, memory-bounded physical choice for each proper-subset coverage-miss request, forced-route payload equivalence, calibrated cost/crossover evidence, unchanged route-independent CPU and packed-batch reuse, and a repeat of the affected Slice 13 interaction matrix. Record those results as a later loading-optimization comparison; do not mark adaptive routing implemented merely because the interaction milestone has passed.

## Conclusion

The original cache-to-canvas investigation identified three bottlenecks:

1. **Warm and cold rendering: primary problem**

   Thousands of independent VisPy nodes cause a 24-second first draw and approximately one-second warm frames.

2. **Cold point reads: secondary major problem**

   Sparse AAMP ranges trigger 4,291 chunk decodes across 69 serial bucket batches, taking approximately 4.14 seconds.

3. **Quadratic residency bookkeeping: avoidable implementation defects**

   CPU retention adds approximately 852 ms; GPU retention adds approximately 1.84 seconds.

The following priorities include the earlier implemented fixes. The next unimplemented interaction work is Slice 12, followed by the initial Slice 13 baseline; Slice 11 is explicitly deferred.

1. Replace one-visual-per-logical-tile rendering with one snapshot visual/program and one VBO fed by worker-prepared immutable render batches. This removes the quadratic GPU residency path rather than optimizing a tile-resource design that is no longer needed. Preserve logical tiles only at the storage and CPU-residency boundaries, and keep tile-proportional packing off the GUI thread. Add a second VBO only if measured behavior justifies ping-pong storage.
2. Fix the quadratic CPU residency path, which remains useful for reusing decoded logical tiles across viewport requests.
3. Stop reading point-level `value_id` on range-resolved and value-major proper-subset paths; construct aligned IDs from the selected values and intervals. A later complete-tile filtering route may decode point-level IDs only when its measured total physical cost is lower.
4. Make a location-only value-major sidecar for every serialized level a mandatory part of the new cache schema alongside the existing tile-major payload. This deliberately duplicates all-level location bytes, projected at approximately 1.09 GiB and a 69% cache increase, while reusing the manifest and value-to-tile catalog and omitting duplicate point-level `value_id` and `point_id` arrays. Do not implement backward compatibility: pre-change tile-major-only or partially covered caches must be rebuilt. Choose LOD first, initially route all-values and complete-tile reads to tile-major and proper subsets to the mandatory sidecar for the selected level, then remove eager retention of the complete 568.4 MiB sparse bucket lookup from the viewer runtime without replacing it with a fallback-index cache. Keep the persisted ranges initially for construction and validation. Measure actual compressed size, cold and warm selected-value wall time, decoded bytes, physical operations, startup, and peak lookup memory.
5. Implement Slice 12 next: replace exact-viewport payload lifetime with stable, tile-aligned render coverage after LOD selection. Reuse a complete selected-level payload when it fits; otherwise grow deterministic bounded coverage around the visible tiles, add measured LOD hysteresis, and retain at most the current and immediately previous immutable CPU batches. Keep one GPU VBO and send only nonresident coverage tiles through the existing fixed routes. Adaptive routing is not required to avoid repeated planning, packing, and VBO replacement on active coverage hits. Previous packed-batch hits likewise avoid reading and packing, but may require uploading a different payload into that VBO.
6. Establish the initial Slice 13 all-level interaction baseline with those routes, including cold coverage construction and dense-subset limitations. Follow with independent Slice 16 selection arming and its affected lifecycle/interaction checks. Add Slice 14 debounce only for measured remaining obsolete dispatch, and accept Slice 15 GPU hardening/scaling or Slice 17 persisted-range removal only at their separate evidence gates; they need not all be implemented before the interaction milestone is complete.
7. Revisit deferred Slice 11 in a later loading-optimization phase, calibrated against the coverage misses that still reach storage. Compare value-major intervals with complete tile-major `location` and `value_id` reads for that miss set, choose one bounded route, and filter tile-major rows in memory when demonstrably cheaper. Keep route-independent CPU/coverage identity, do not use selected-value count alone or restore sparse-range lookup, and extend the Slice 13 report with the crossover and regression evidence.
8. Treat smaller chunks, fewer buckets, and cross-bucket concurrency as secondary comparisons or tuning. The current evidence does not support them as fixes for tile-major sparse decoding or per-tile rendering.

The synthetic 1,000,000-point packing benchmark does not change the current 100,000-point implementation priority. It establishes forward-looking scalability evidence and the additional acceptance work required before the render budget is deliberately increased.

Larger storage tiles would reduce chunks and visual objects, but the benchmark shows that renderer batching can remove the dominant visual-object cost without sacrificing the existing 512-unit spatial read granularity.

This implementation-planning update changes only this roadmap document; it does not implement the slices above.
