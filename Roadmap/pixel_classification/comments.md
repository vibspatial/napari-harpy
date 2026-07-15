I reviewed it. Overall, it’s a strong roadmap: the Phase 1 boundary is clear, the Qt/core separation is sensible, and the reducer/cache compatibility model is unusually thorough. I would resolve a few architectural blockers before implementation.

### Main findings

1. **Pooled class semantics are undefined.**  
   Training pools integer labels from several annotation layers, but nothing guarantees that class `1` means the same thing in every target. Introduce a shared immutable `PixelClassSchema`—ID, name, color, schema ID—used by all annotation layers, classifier metadata, predictions, and saved labels. See the [annotation and classifier contracts](/Users/arne.defauw/VIB/napari_harpy/Roadmap/pixel_classification/pixel_classification_phase_1.md:514).

2. **The per-card cache flow conflicts with cohort-fitted PCA.**  
   Users build caches before annotating or choosing the training scope, but PCA caches depend on the eventual reducer-fit cohort; adding a target then rebuilds every participating cache. See the [top-level flow](/Users/arne.defauw/VIB/napari_harpy/Roadmap/pixel_classification/pixel_classification_phase_1.md:50) and [PCA compatibility rules](/Users/arne.defauw/VIB/napari_harpy/Roadmap/pixel_classification/pixel_classification_phase_1.md:355).  
   My preference for Phase 1 would be:

   - deterministic fixed projection as the default;
   - annotations keyed to the source grid and allowed before cache completion;
   - PCA added only with an explicit global “fit reducer and build cohort caches” workflow.

3. **ConvNeXt feature-grid alignment is missing.**  
   The installed ConvNeXt-Tiny starts with a `4×4`, stride-4 convolution, so its features are not naturally on the source `(y, x)` grid. The roadmap requires those features to be concatenated with full-resolution marker planes but does not define interpolation, pixel-center alignment, or output cropping. The [feature contract](/Users/arne.defauw/VIB/napari_harpy/Roadmap/pixel_classification/pixel_classification_phase_1.md:526) should explicitly specify this.

   Tile extraction also needs a halo/crop contract and a test showing tiled output matches whole-image extraction away from boundaries; otherwise tile seams can affect predictions.

4. **Cache identity does not detect source data changing in place.**  
   URI, element path, shape, axes, and dtype can all remain unchanged while pixel values change, causing incorrect cache reuse. The [cache identity fields](/Users/arne.defauw/VIB/napari_harpy/Roadmap/pixel_classification/pixel_classification_phase_1.md:320) need a source revision/fingerprint policy, including a policy for in-memory data where no durable source identity exists.

5. **Canonical IDs should not be used directly as directory names.**  
   IDs contain colons, while the layout uses them as Zarr path components. That breaks directory-backed stores on Windows despite the project being OS-independent. Keep the full canonical ID in the manifest, but use the hexadecimal digest or another filesystem-safe encoding as the Zarr key. See the [ID format and layout](/Users/arne.defauw/VIB/napari_harpy/Roadmap/pixel_classification/pixel_classification_phase_1.md:238).

6. **Coordinate systems may cause enormous duplicate caches.**  
   The roadmap requires a physical cache per coordinate-system target and includes the coordinate system in `cache_id`. If the same image element belongs to several coordinate systems, its intrinsic pixel features are identical; only its viewer transformation differs. Given the projected multi-gigabyte cache sizes, cache identity should normally be based on the source image grid, with coordinate-system binding stored separately in target/prediction metadata. See [cache separation](/Users/arne.defauw/VIB/napari_harpy/Roadmap/pixel_classification/pixel_classification_phase_1.md:250).

7. **Some Phase 1 contracts remain ambiguous.**

   - Explicitly support only `(c, y, x)`/`(y, x)`, or define what happens with `z`, time, and other dimensions.
   - Define exactly how a viewer layer determines the “active target.”
   - Define ConvNeXt’s multiplex-channel strategy—independent grayscale-to-RGB passes, selected three-channel groups, or another policy.
   - Define writable/read-only/in-memory behavior when saving predictions.

8. **A few smaller inconsistencies should be cleaned up.**

   - Annotation is Slice 3 but depends on cache readiness, while the cache store arrives in Slice 5.
   - “Rebuild with a new cache ID” conflicts with identical compatible requests producing the same semantic ID.
   - Rejecting an “unbalanced” training scope is underspecified and probably too strict; warn or balance samples, while rejecting only empty and one-class scopes.
   - The full-suite commands at [lines 713–718](/Users/arne.defauw/VIB/napari_harpy/Roadmap/pixel_classification/pixel_classification_phase_1.md:713) conflict with this repository’s focused-test instructions.

My overall recommendation is to keep the roadmap, but revise the workflow around a shared class schema, source-keyed annotations, and deterministic random projection for Phase 1. Those decisions remove most of the circular state and cache-rebuild complexity without weakening the user-facing feature. No files were changed.