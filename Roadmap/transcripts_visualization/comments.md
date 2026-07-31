Yes—we can move to implementation. The core contracts in both roadmaps are now consistent enough to begin V0, followed by V1.

The roadmaps agree on:

- the fresh `core/multi_scale_cache_points/` package;
- explicit physical Parquet resolution without inspecting Dask graphs;
- `x`, `y`, and generic categorical `value`, defaulting to the physical `gene` column;
- validation returning `ValidatedPointsSource` directly;
- deterministic dataset-relative fragment ordering;
- Harpy-generated `uint64 point_id`;
- footer-first validation followed by one bounded content scan;
- no validation report, progress protocol, cancellation, or caller-supplied identity;
- benchmark measurements remaining outside the validation result.

The initial V0 scope is particularly clear and ready to implement: [`PointColumnSelection`, `ParquetPointsSource`, minimal errors, narrow exports, and tests](/Users/arne.defauw/VIB/napari_harpy/Roadmap/transcripts_visualization/validation_cache_29_7_26.md:387). The existing implementation also confirms that the planned V1 operations—checking backed state, using `locate_element`, and verifying a Dask dataframe—are viable with the current SpatialData API ([existing resolver](/Users/arne.defauw/VIB/napari_harpy/src/napari_harpy/_transcript_tiles.py:188)).

Before V2–V5 are frozen, I found four small issues worth resolving.

1. `ValidationEvidence` is still dangling

The validation outcome still promises evidence ([line 47](/Users/arne.defauw/VIB/napari_harpy/Roadmap/transcripts_visualization/validation_cache_29_7_26.md:47)), and an evidence enum remains ([line 257](/Users/arne.defauw/VIB/napari_harpy/Roadmap/transcripts_visualization/validation_cache_29_7_26.md:257)). However, `ValidatedPointsSource` contains no evidence fields ([line 284](/Users/arne.defauw/VIB/napari_harpy/Roadmap/transcripts_visualization/validation_cache_29_7_26.md:284)), consistently with the decision not to expose a diagnostics report.

I recommend removing `ValidationEvidence` and the remaining evidence requirements. The roadmap can simply state which facts are authoritative:

- row count and fragment offsets: Parquet metadata;
- bounds and value counts: streaming scan;
- stale-source detection: source signature.

2. Row-group signature material needs an internal model

`ParquetSourceFragment` currently retains only `row_group_count` ([line 228](/Users/arne.defauw/VIB/napari_harpy/Roadmap/transcripts_visualization/validation_cache_29_7_26.md:228)), while the signature requires each row group’s row count and compressed size ([line 607](/Users/arne.defauw/VIB/napari_harpy/Roadmap/transcripts_visualization/validation_cache_29_7_26.md:607)).

Before V2, I recommend defining an internal immutable row-group record, probably:

```python
@dataclass(frozen=True)
class ParquetSourceRowGroup:
    row_count: int
    compressed_size_bytes: int
```

`ParquetSourceFragment` can then contain `row_groups: tuple[ParquetSourceRowGroup, ...]`. It need not be publicly exported.

3. Source mutation during validation/build is unspecified

The signature captures the inventory, but the roadmap does not currently require checking that the files remain unchanged while the content scan runs.

A practical policy would be:

- V5 rechecks the footer inventory/signature after the streaming scan;
- validation fails if it differs from the pre-scan signature;
- Phase 1 rechecks the source signature before publishing the completed cache.

This does not turn the signature into a full content hash, but it closes the ordinary “source changed while processing” race.

4. Gate C is sequenced incorrectly

Gate C is marked “after V5” but requires measured Xenium performance ([line 1011](/Users/arne.defauw/VIB/napari_harpy/Roadmap/transcripts_visualization/validation_cache_29_7_26.md:1011)); performance is measured in V6, and Phase 0’s definition of done also requires V6 ([line 1036](/Users/arne.defauw/VIB/napari_harpy/Roadmap/transcripts_visualization/validation_cache_29_7_26.md:1036)).

I recommend:

- Gate C after V5: approve the validation and error contracts;
- Gate D after V6: approve measured performance and readiness for the exact writer.

One later, non-blocking cleanup: the parent roadmap still asks to benchmark both 256- and 512-unit exact tiles ([line 1455](/Users/arne.defauw/VIB/napari_harpy/Roadmap/transcripts_visualization/multi_tile_cache_29_7_26.md:1455)), despite locking the initial schedule to a 512-unit exact level ([line 1470](/Users/arne.defauw/VIB/napari_harpy/Roadmap/transcripts_visualization/multi_tile_cache_29_7_26.md:1470)). I would benchmark 512 by default and use 256 only as a contingency if 512 exposes a concrete writer/layout problem.

One scope boundary is already consistent but worth emphasizing: `value` is semantically generic, but V1 supports only string-valued categorical columns ([line 82](/Users/arne.defauw/VIB/napari_harpy/Roadmap/transcripts_visualization/validation_cache_29_7_26.md:82)). It does not initially mean arbitrary numeric or binary values.

My conclusion: start V0 now. None of the remaining issues affects its models or tests. I would resolve the evidence and row-group-model points before implementing V2, and settle the mutation check and review-gate wording before V5. No files were changed during this review.