

The biggest unresolved design choice is how coarse levels are constructed and sampled. In the cache draft, coarse levels are rebuilt directly from the exact source dataframe with deterministic hash sampling (Transcripts_Tile_cache.md (line 344), Transcripts_Tile_cache.md (line 354)). In the broader design note, coarse levels are described as bottom-up from child tiles with a preference for spatially stratified sampling (visualizing_transcripts.md (line 203), visualizing_transcripts.md (line 216)). Those are materially different implementations, so I would pick one explicitly in this doc.

The current cache schema does not support the future “rare gene stays exact at all zoom levels” behavior described in the viewer note. The runtime note says exact mode for small gene selections (visualizing_transcripts.md (line 321)), but both docs also say gene filtering happens after tile load (Transcripts_Tile_cache.md (line 480), visualizing_transcripts.md (line 354)). With no per-tile gene index, exact-mode on a whole-slide view would still require reading huge exact tiles. I’d either remove that future promise or mark it as requiring a later schema extension.

For “billions of points,” the build cost model is still a bit underspecified. The draft currently implies one full pass over the source plus tile grouping for the finest level, then another full pass per coarse level from the original dataframe (Transcripts_Tile_cache.md (line 331), Transcripts_Tile_cache.md (line 348)). That may be acceptable as offline preprocessing, but the doc should say so explicitly, or switch to a bottom-up build to avoid repeated full-data scans.

max_points_per_tile currently means two different things: shard size for dense exact tiles and sample budget for coarse tiles (Transcripts_Tile_cache.md (line 52), Transcripts_Tile_cache.md (line 339), Transcripts_Tile_cache.md (line 349)). I’d separate those into two knobs unless you intentionally want them locked together.

The SpatialData helper hardcodes Path(sdata.path) / "points" / points_key / "transcripts_vis" (Transcripts_Tile_cache.md (line 102)). In this repo, _persistence.py already resolves element paths via SpatialData metadata rather than assuming the on-disk path shape (src/napari_harpy/_persistence.py (line 322)). I’d do the same for points elements, otherwise the helper is brittle to storage-layout changes.

Input edge cases still need explicit behavior. The validation section covers missing columns and bad parameter values, but not empty dataframes, null gene values, or NaN/inf coordinates (Transcripts_Tile_cache.md (line 267)). Those cases affect bounds, tile assignment, and deterministic rebuilds, so I’d define “reject” vs “drop” now.

“Deterministic hash sampling” is still too vague for implementation. If that uses Python’s built-in hash, it will not be stable across runs. The doc should name a stable hash mechanism for both transcript_id and fallback (x, y, gene_id) sampling (Transcripts_Tile_cache.md (line 354)).

leaf_tile_size needs a one-line clarification that it is in stored coordinate units, not screen pixels. The cache draft implies data-space units (Transcripts_Tile_cache.md (line 50), Transcripts_Tile_cache.md (line 239)), while the runtime note sometimes talks in “px” terms (visualizing_transcripts.md (line 380)). That is easy to fix, but worth making explicit.

metadata.json is missing cache provenance. Right now it stores geometry, but not enough to tell whether the cache is stale after points.parquet changes (Transcripts_Tile_cache.md (line 128)). At minimum I’d consider storing source row count, source element key/path, chosen column names, and the build parameters.





Spec Items To Tighten Before Proceeding


TranscriptTileCache contract

phase_1_spatial_first_cache.md (line 127) says “level metadata” but does not list exact dataclass fields. The implementation follows the older full plan, but Phase 1 should explicitly say whether level metadata means only n_levels, finest_level, leaf_tile_size, or whether we need per-level paths/metadata too.



output_path semantics

Specify that output_path passed to build_transcript_visualization_cache(...) is the final transcripts_vis/ directory, not the points-element directory. Also specify whether returned paths should preserve the input path or be resolved/absolute.



Manifest path conventions

The roadmap should say whether manifest.level_file stores level_2.parquet, levels/level_2.parquet, or another relative path. That matters for tests and the future reader.



Exact Parquet dtypes

Step 6 says to standardize dtypes, but the phase doc should list them concretely for tile_x, tile_y, level, row_group, tile_shard, x_rel, y_rel, and gene_id.



Validation behavior under Dask

The doc already says to reject empty inputs, null genes, and invalid coordinates. It should specify whether validation performs an eager Dask reduction/compute, and whether this should be combined with the bounds computation to avoid duplicate passes.



Gene missing-value policy

“Reject missing gene values if ambiguous” should become exact: reject None, pandas NA, and NaN before string coercion. Also decide whether empty strings are valid genes.



tile_id format

Step 6 says compute tile_id, but Phase 1 should explicitly define it, probably f"{level}/{tile_x}/{tile_y}", matching the parent plan.



Deterministic coarse sampling details

Before implementing coarse levels, specify the exact stable hash/digest algorithm, transcript-id encoding rules, null transcript-id behavior, and tie-break ordering. Also clarify that coarse_tile_budget applies globally per coarse tile, not per Dask partition.



tile_shard determinism

Since finest-level writing may be partition-local, define how tile_shard is assigned when the same tile appears in multiple partitions. Otherwise rebuilds can be correct but hard to test deterministically.



Metadata provenance

Current metadata.json stores geometry only. Decide now whether Phase 1A should also store source column names, source row count, build parameters, and/or source points path. If not, explicitly defer stale-cache/provenance detection.



Dependency update

The phase doc says direct dependencies for dask[dataframe] and pyarrow “if still needed”; pyproject.toml does not currently include them. Since the planned public API and writer require both, this should be made non-optional for Phase 1A.