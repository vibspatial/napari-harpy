# Sampling artefacts at tile boundaries — 17 September 2026

## Summary

The current sampled cache levels can introduce rectangular changes in apparent transcript density. The mechanism is a fixed representative quota applied independently to each logical tile: two tiles with different original point counts can retain the same number of representatives, giving them different sampling fractions.

This is especially problematic where only part of a tile contains transcripts. Its quota is concentrated in the occupied region, which can then appear denser than an equally dense region in a fully occupied neighbouring tile.

The screenshots are consistent with this mechanism. Inspection of the construction code, a read-only check of the cache manifest, and a synthetic run of the actual sampler confirm that the current policy can produce it. These checks do not establish that every visible boundary in the screenshots has this single cause.

**Status:** investigation findings and recommended direction; no sampling implementation change or cache rebuild has been performed.

## 1. Observations from the UI

The supplied screenshots show rectangular density discontinuities, particularly around tissue boundaries and partially occupied regions. Both distinct views display **Bridge**, with **All values** selected:

- 328,900 rendered points across 90 tiles, at a 1.5-pixel point diameter.
- 2,534,220 rendered points across 624 tiles, at a 1.0-pixel point diameter.

Thus the issue already appears at the first sampled level; it does not require spatial aggregation into larger tiles. The different point diameters and viewports mean these screenshots are not a controlled quantitative comparison of brightness.

Raising the UI render budget can make a finer level eligible, but does not change the membership of an already constructed Bridge or Spatial level. Hysteresis changes when a level is selected; it cannot repair density distortion stored in that level.

## 2. Confirmed construction behaviour

### Independent per-tile quotas

Exact retains all source points. In the inspected cache, Bridge shares Exact's 512-unit tile geometry and retains:

```text
Bridge points in tile T = min(Exact points in T, 4,096)
```

The [Bridge writer](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/core/multi_scale_cache_points_zarr/writer/bridge.py:249) plans this count and supplies the per-tile capacity to the sampler. The 4,096-point construction quota is separate from the viewport's runtime render budget.

For an Exact tile containing N points and a construction capacity K, the retained fraction is:

```text
r(T) = min(1, K / N)
```

That fraction varies between tiles. Dense tiles lose a larger proportion of their points, while sparse tiles retain more, potentially all of them.

The sampler never invents or duplicates points to fill a quota: when N <= K, it [retains every candidate](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/core/multi_scale_cache_points_zarr/sampling.py:118). The problem is unequal retention fractions, not upsampling.

### Why the 16 × 16 microgrid does not prevent it

Within a tile, the sampler distributes its quota proportionally to the number of candidates in each microgrid cell, with integer largest-remainder allocation. See [the allocation implementation](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/core/multi_scale_cache_points_zarr/sampling.py:197).

This approximately preserves relative density **within one tile**. It does not coordinate the sampling fraction **between tiles**. Empty cells receive no representatives; the fixed tile quota is allocated among the cells containing candidates. Partial occupancy therefore concentrates the retained points into a smaller area.

### Coarser levels can inherit and add distortion

The [Spatial writer](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/core/multi_scale_cache_points_zarr/writer/spatial.py:566) assembles candidates from the immediate finer level and samples them under another per-tile capacity. These candidates have already been sampled; unequal retention introduced earlier is not automatically corrected. Applying another independent tile quota can introduce further density differences.

Fixing Bridge alone while retaining the same policy at subsequent Spatial levels would therefore be incomplete.

## 3. Reproduced example using the current sampler

A synthetic check supplied the actual `_select_sampled_tile_indices()` function with two 512 × 512 tiles. Points were arranged at the same original density wherever occupied: a regular grid with spacing 2 units horizontally and 4 units vertically. Tile A contained that grid throughout; tile B contained it only within a 128 × 512 strip. Both used the Bridge capacity of 4,096.

| Tile | Occupied area | Input points | Sampled points | Retained fraction |
|---|---:|---:|---:|---:|
| A: fully occupied | 512 × 512 | 32,768 | 4,096 | 12.5% |
| B: quarter occupied | 128 × 512 | 8,192 | 4,096 | 50% |

The input density in both occupied regions was **0.125 points per square intrinsic unit**. After sampling:

- Tile A: **0.015625** points per square intrinsic unit.
- Tile B's occupied strip: **0.0625** points per square intrinsic unit.

The occupied strip becomes **four times denser** than tile A, despite identical original density. This demonstrates a systematic sampling effect, not merely random variation in representative selection.

## 4. Check against the existing cache

A read-only manifest check was performed on:

```text
/Users/arne.defauw/VIB/DATA/test_data/
  sdata_xenium_full_data_core.zarr/points/
    transcripts_global_ROI1/transcripts_vis_zarr
```

The cache records the `harpy-value-neutral-stratified-splitmix64-v1` sampling method. It contains **7,294 Bridge tiles**, each with exactly `min(Exact count, 4,096)` points. **4,123 tiles** have more than 4,096 Exact points and are therefore capped.

This confirms that the investigated cache uses the policy described above. The check compared counts for matching Exact/Bridge tile coordinates; it did not measure occupied tissue area or register screenshot boundaries against tile coordinates.

## 5. Recommended direction: preserve density across tile boundaries

If the desired visualization should preserve relative transcript density, investigate a **common sampling fraction for each level**, applied consistently across tiles, rather than independently filling each tile's quota.

For example, a common 12.5% fraction would retain approximately:

| Tile | Input points | Representatives at 12.5% |
|---|---:|---:|
| A | 32,768 | 4,096 |
| B | 8,192 | 1,024 |

The occupied regions would then have the same expected sampled density. Counts are approximate for a probabilistic/hash-threshold implementation; an exact allocation scheme would need an explicit rounding policy.

Storage can remain tiled. Tile boundaries should organize addressing and IO without independently normalizing the visible density. Deterministic point-ID priorities with level-wide thresholds are one candidate implementation, not a settled algorithm. Preserve value-neutral selection and deterministic membership; consider nested membership between levels to avoid unnecessary representative changes.

Do not simply apply the old per-tile cap after common-fraction sampling: wherever that cap binds, sampling fractions would again differ and the artefact could return. Increasing the existing tile quota or changing rendering opacity may alter visibility of the problem, but does not remove its underlying cause.

## 6. Decisions required before implementation

- **Level fractions and terminal budget:** choose the level schedule and determine how the complete overview is guaranteed to fit its construction budget. An expected sample count alone is not a strict upper-bound guarantee.
- **Construction bounds:** preserve bounded reads, writes, and temporary memory without relying on density-normalizing per-tile quotas. Physical chunk/bucket limits and logical sampling policy serve different purposes.
- **Hierarchy:** decide whether each level is derived from Exact or from a consistently sampled finer level, and how overall inclusion fractions remain correct across the hierarchy. Further uniform thinning of the existing biased Bridge cannot recover discarded points or undo its unequal sampling fractions.
- **Rounding and sparse regions:** retain determinism without systematically topping up small tiles or occupied cells. Such top-ups can reintroduce density bias. Preserve honest reporting when sampled levels omit selected values.
- **Cache contracts:** update planned counts/bounds, sampler metadata, writers, and validation together. Existing checks expecting `min(candidate_count, capacity)` must change with the sampling contract. Regenerate the affected sampled tile-major payloads, value-major copies, and lookup/count metadata consistently; a runtime-only change cannot repair the current cache.

These are follow-up design decisions, not approval to rebuild or change the cache during this investigation.

## 7. Suggested validation

1. Keep the full-tile versus quarter-occupied example as a focused regression case: equal original density should not become systematically unequal because of occupancy or tile boundaries.
2. Test a uniformly dense shape crossing tile boundaries and a genuinely nonuniform-density example. Removing seams must not flatten real spatial density differences.
3. Check Bridge and multiple Spatial levels, including partial edge tiles, empty space, sparse values, and both all-values and selected-value views.
4. Verify deterministic/value-neutral membership, correct counts and metadata, equivalent logical results from both physical orderings, and the terminal overview budget.
5. Compare UI images at matched viewport, level, diameter, and opacity. Measure construction time, peak RSS, cache size, and viewport preparation/draw costs separately: a better sampling distribution can change payload counts and tile occupancy.

The acceptance goal is to remove systematic tile-boundary density discontinuities while preserving meaningful spatial variation, deterministic construction, and explicit resource limits—not to make every region look equally dense.
