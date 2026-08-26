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

Smaller chunks greatly reduce decoded content without changing physical row order. They do not reduce the number of sparse selections, and they increase the number of independently handled inner chunks from 4,291 to 5,305 in the 64-row case. They also leave all 1,067 current shards involved because AAMP is distributed throughout the full tissue. Smaller `location` chunks are therefore a useful intermediate experiment, but they do not provide the locality of a value-major payload.

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
| One or a small subset of values | Per-level value-major coordinate payload |

The cache catalog already persists value-to-tile records in `(level, value_id, manifest_index)` order. A value-major coordinate payload can follow that same record order and use compact per-level/value point pointers plus the existing record counts for exact addressing.

For a proper selected-value request, point-level `value_id` does not need to be stored in or read from the value-major payload: the requested value and the catalog intervals already identify it. A full-extent AAMP read would become one contiguous value interval; a rectangular partial viewport would normally become a small number of spatial runs within that value instead of one interval per positive tile.

This design deliberately spends cache-construction time and storage to improve runtime behavior. It also creates a path that does not require the complete 568.4 MiB bucket sparse-range lookup to be resident for selected-value reads. The existing small selected-value catalog index can perform discovery, while the value-major payload provides direct coordinate access.

### Lower-risk cache improvements

Before or independently of the dual-layout work, two smaller changes are justified:

1. **Do not read point-level `value_id` for proper subsets.**

   `resolve_selected_tile_intervals()` already knows the selected value associated with each range. Reconstructing the aligned IDs from the resolved ranges would remove the measured 1.86-second `value_id` Zarr boundary for AAMP. A one-value renderer could alternatively use a uniform value ID.

2. **Give `location` an independent, smaller chunk-row setting.**

   A 128- or 256-row prototype would reduce AAMP coordinate decode amplification substantially without forcing construction-only `point_id` storage to use the same very small chunks. The benchmark must include cache size, inner-chunk index size, physical reads, and wall time; decoded-row reduction alone is not sufficient acceptance evidence.

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

A two-buffer ping-pong variant can retain atomic activation semantics: prepare and upload the pending snapshot into an inactive VBO, then swap which buffer is visible. This still bounds the renderer to two visuals and two VBOs rather than one resource per logical tile.

Tile offsets should be folded into the assembled positions relative to a snapshot-local or batch-local origin. Precomposing that origin into the root transform preserves the existing protection against float32 precision loss from large absolute coordinates.

The shared palette texture remains applicable. Logical CPU tile residency can also remain unchanged so warm pan and zoom requests reuse decoded data. Per-logical-tile GPU residency is no longer needed; if profiling later justifies GPU-side page reuse, it should use a small fixed pool of larger pages rather than thousands of scene nodes.

This design directly addresses all three renderer symptoms:

- resource creation becomes constant in visual count;
- first draw performs one or two VBO uploads and shader/program preparations;
- warm frames issue one active point draw instead of thousands of draw submissions.

It also makes the GPU budget representative of the resources it controls. The current logical vertex-byte accounting does not capture per-visual Python, program, scene-node, VBO-wrapper, and driver overhead.

## Conclusion

There are three proven bottlenecks:

1. **Warm and cold rendering: primary problem**

   Thousands of independent VisPy nodes cause a 24-second first draw and approximately one-second warm frames.

2. **Cold point reads: secondary major problem**

   Sparse AAMP ranges trigger 4,291 chunk decodes across 69 serial bucket batches, taking approximately 4.14 seconds.

3. **Quadratic residency bookkeeping: avoidable implementation defects**

   CPU retention adds approximately 852 ms; GPU retention adds approximately 1.84 seconds.

The practical priority is therefore:

1. Fix the quadratic CPU and GPU residency paths.
2. Replace one-visual-per-logical-tile rendering with one active snapshot VBO or a two-buffer ping-pong renderer. Preserve logical tiles only at the storage and CPU-residency boundaries.
3. Stop reading point-level `value_id` for proper selected-value requests; reconstruct it from the already resolved value ranges or represent a one-value snapshot with a uniform.
4. Prototype a per-level value-major coordinate payload alongside the existing tile-major payload. Compare it with 128- and 256-row `location` chunks using cold-read wall time, decoded bytes, physical operations, cache size, and startup memory.
5. Treat fewer buckets and cross-bucket concurrency as secondary tuning. The current evidence does not support them as fixes for tile-major sparse decoding or per-tile rendering.
6. Add viewport debounce to avoid starting expensive cold requests for transient zoom states after the underlying read and render costs are controlled.

Larger storage tiles would reduce chunks and visual objects, but the benchmark shows that renderer batching can remove the dominant visual-object cost without sacrificing the existing 512-unit spatial read granularity.

This follow-up investigation changed only this roadmap document. No repository source code was changed.
