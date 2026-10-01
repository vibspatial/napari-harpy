**The dual layout is a sound design for large-scale transcript viewing, and I would keep it.** It addresses the two main access patterns well: spatial browsing across all genes, and inspecting selected genes across a large tissue. The remaining limitations concern viewport planning, buffer reuse, and access to full detail in dense regions.

I reviewed commit [`c055c48`](https://github.com/vibspatial/napari-harpy/commit/c055c4829d41142cb6f0a6a607c7284c3a4db6ec), including the writer, reader, sampling, worker, and VisPy integration. I also ran isolated checks using the existing planner methods. I did **not** run a complete tissue-to-OpenGL benchmark, so performance conclusions below distinguish code behavior from measured rendering speed.

**The two physical layouts complement each other well.**

| Representation | Current use | Benefit |
|---|---|---|
| Tile-major buckets | All genes in intersecting tiles | Spatially nearby points can be read together |
| Value-major arrays, one per level | Any proper subset of genes | A gene’s coordinates are grouped together across the tissue |
| Shared manifest and value-to-tile catalog | Planning both routes | Identifies relevant tiles and coordinate intervals without reading points |

This routing is implemented in [`plan_viewport()`](https://github.com/vibspatial/napari-harpy/blob/c055c4829d41142cb6f0a6a607c7284c3a4db6ec/src/napari_harpy/core/multi_scale_cache_points_zarr/reader.py#L941).

The key benefit comes from **physically reordering coordinates**. An additional gene index into tile-major storage would locate the right rows, but those rows would still be scattered across compressed chunks. Your value-major representation changes that physical access pattern.

The duplication is also reasonably economical: the sidecar stores coordinates and per-value pointers, while reconstructing gene IDs during reading. From the [schema](https://github.com/vibspatial/napari-harpy/blob/c055c4829d41142cb6f0a6a607c7284c3a4db6ec/src/napari_harpy/core/multi_scale_cache_points_zarr/CACHE_FORMAT.md#L66), the raw point arrays cost approximately **20 bytes per tile-major point plus 8 bytes for its value-major copy**, excluding indexes and compression. That is a 40% increase in those raw point arrays; the compressed percentage will differ.

**The renderer is already architecturally appropriate for this cache.**

It uses a custom layer with one combined vertex buffer and a palette texture. Each vertex occupies 12 bytes: two coordinates and one gene index. At the default 100,000-point ceiling, that is approximately **1.2 MB of vertex payload**. Palette changes avoid coordinate uploads. The worker owns cache access, and the coordinator rejects stale results. These are useful foundations for responsive viewing. [Vertex contract](https://github.com/vibspatial/napari-harpy/blob/c055c4829d41142cb6f0a6a607c7284c3a4db6ec/src/napari_harpy/viewer/tiled_points/contracts.py#L19), [renderer](https://github.com/vibspatial/napari-harpy/blob/c055c4829d41142cb6f0a6a607c7284c3a4db6ec/src/napari_harpy/viewer/tiled_points/vispy/visuals.py#L178), [coordinator](https://github.com/vibspatial/napari-harpy/blob/c055c4829d41142cb6f0a6a607c7284c3a4db6ec/src/napari_harpy/viewer/tiled_points/runtime/coordinator.py#L46).

This makes browsing datasets containing hundreds of millions—or potentially a billion—transcripts plausible through bounded visible subsets. Total dataset size still affects indexes, construction, and storage access.

The most important findings are:

1. **Dense tiles can prevent full-resolution viewing, regardless of zoom.**

   Level selection counts **every point in each intersecting tile**, including points outside the viewport. If one Exact tile exceeds the effective point budget, zooming further inside it never makes Exact eligible. [Selection code](https://github.com/vibspatial/napari-harpy/blob/c055c4829d41142cb6f0a6a607c7284c3a4db6ec/src/napari_harpy/core/multi_scale_cache_points_zarr/reader.py#L1166).

   I reproduced this with an Exact tile containing 200,000 points and a Bridge tile containing 4,096. With a 100,000-point budget, viewports with side lengths of 400, 10, and 0.01 coordinate units all selected Bridge.

   **This is the most consequential limitation for inspecting individual transcripts.** Smaller leaf tiles help, but robust access to full detail needs finer spatial subdivision or a bounded Exact read-and-clip path. Tile sizes should reflect peak local density and source coordinate units.

   There is a related overview problem: the coarsest stored level can exceed the runtime budget. For example, a 100,000-point overview exceeds the default density budget on a 1,000 × 800 canvas: approximately 88,888 points. The runtime then retains the previous view. A further point-based thinning fallback would make overview rendering more reliable. [Budget calculation](https://github.com/vibspatial/napari-harpy/blob/c055c4829d41142cb6f0a6a607c7284c3a4db6ec/src/napari_harpy/viewer/tiled_points/napari/viewport.py#L129), [over-budget behavior](https://github.com/vibspatial/napari-harpy/blob/c055c4829d41142cb6f0a6a607c7284c3a4db6ec/src/napari_harpy/viewer/tiled_points/runtime/composition.py#L274).

2. **CPU tile reuse still leads to complete buffer preparation and upload.**

   Even when all requested tiles are resident, the worker assembles and packs the complete snapshot. Each accepted nonempty snapshot then calls `VertexBuffer.set_data(..., copy=True)`. [Worker](https://github.com/vibspatial/napari-harpy/blob/c055c4829d41142cb6f0a6a607c7284c3a4db6ec/src/napari_harpy/viewer/tiled_points/runtime/cache_session.py#L557), [upload staging](https://github.com/vibspatial/napari-harpy/blob/c055c4829d41142cb6f0a6a607c7284c3a4db6ec/src/napari_harpy/viewer/tiled_points/vispy/visuals.py#L238).

   The single-buffer design avoids thousands of draw calls, which is valuable. I would first add reuse of an already prepared and uploaded batch when the new view is contained within its coverage, with unchanged selection, level, and valid budgets.

   **There is a documentation mismatch here:** [Slice 12a says this is implemented](https://github.com/vibspatial/napari-harpy/blob/c055c4829d41142cb6f0a6a607c7284c3a4db6ec/Roadmap/transcripts_visualization/comments_slow_rendering_26_8_26.md#L1968), but the corresponding retained-viewport implementation and named benchmark script are absent from this commit. I would reconcile that before attributing its reported timings to this branch.

3. **Viewport planning still performs work proportional to offscreen metadata.**

   `_visible_manifest_rows()` scans every manifest tile at each evaluated level. Selected-gene matching also scans the selected genes’ records across the level and allocates a level-sized lookup array. [Spatial lookup](https://github.com/vibspatial/napari-harpy/blob/c055c4829d41142cb6f0a6a607c7284c3a4db6ec/src/napari_harpy/core/multi_scale_cache_points_zarr/reader.py#L1329), [selected-gene matching](https://github.com/vibspatial/napari-harpy/blob/c055c4829d41142cb6f0a6a607c7284c3a4db6ec/src/napari_harpy/core/multi_scale_cache_points_zarr/reader.py#L1647).

   This can be inexpensive with thousands of tiles, but becomes increasingly relevant as tiles get smaller—the natural response to the dense-tile problem.

   I would use grid-coordinate bounds and row-based searches to retrieve intersecting tiles, then intersect against sorted gene records. Also, value-major reading currently recomputes cumulative offsets across a gene’s complete level records; caching those offsets for the committed selection would avoid repeated work. [Offset reconstruction](https://github.com/vibspatial/napari-harpy/blob/c055c4829d41142cb6f0a6a607c7284c3a4db6ec/src/napari_harpy/core/multi_scale_cache_points_zarr/reader.py#L1780).

4. **The fixed routing policy will have an inefficient corner case.**

   Every proper gene subset uses value-major storage. That is attractive for a few genes across a large area. For almost all genes in a small area, complete tile-major reads followed by filtering may touch fewer chunks.

   The dual layout already gives you both options. A later routing decision should compare estimated physical reads and decoded bytes **after excluding resident tiles**. The number of selected genes alone is insufficient: gene abundance and viewport size matter too. This is an optimization opportunity, not a reason to change the format.

5. **The sampled hierarchy is useful for navigation, with scientific interpretation limits.**

   Exact → same-grid Bridge → progressively larger spatial tiles is a sensible hierarchy. Deterministic, gene-neutral sampling preserves actual point locations and avoids gene-order-dependent selection. [Hierarchy](https://github.com/vibspatial/napari-harpy/blob/c055c4829d41142cb6f0a6a607c7284c3a4db6ec/src/napari_harpy/core/multi_scale_cache_points_zarr/build_plan.py#L1), [sampling](https://github.com/vibspatial/napari-harpy/blob/c055c4829d41142cb6f0a6a607c7284c3a4db6ec/src/napari_harpy/core/multi_scale_cache_points_zarr/sampling.py#L25).

   However, fixed per-tile caps flatten density differences between saturated tiles, and rare genes can disappear from sampled levels. Omission reporting helps, but does not describe every local sampling loss. Quantitative counts, absence claims, and QC selections should resolve against Exact data.

6. **The current implementation supports gene-colored 2D viewing, with point inspection still to build.**

   Value-major payloads lack point identity and per-transcript attributes, and `_get_value()` deliberately provides no picking. Consequently, hover details, persistent transcript selection, quality filtering, and selection export need additional contracts. [Current picking boundary](https://github.com/vibspatial/napari-harpy/blob/c055c4829d41142cb6f0a6a607c7284c3a4db6ec/src/napari_harpy/viewer/tiled_points/napari/layer.py#L263).

   Aligned point IDs and frequently used attributes in both physical orders are a natural extension.

One detail I would **keep** is tile-relative coordinates. Reconstructing positions adds offsets into a contiguous output array; it does not require a large spatially indexed array or random global writes. The separate value-major-to-tile regrouping does incur copies, but changing to absolute coordinates would not remove that regrouping. [Packing implementation](https://github.com/vibspatial/napari-harpy/blob/c055c4829d41142cb6f0a6a607c7284c3a4db6ec/src/napari_harpy/viewer/tiled_points/render_batch.py#L21).

My implementation priorities would be **full-detail access in dense regions, retained-buffer reuse, then viewport-local metadata planning**. After those, benchmark adaptive routing using sparse and dense gene selections, small and full views, and repeated LOD transitions. Measure time to the updated image, GUI frame gaps, decoded bytes, and peak memory; storage read time alone will not establish smooth interaction.