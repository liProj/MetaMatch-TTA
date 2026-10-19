# Included experiment records

`final/final.json`, `final/final_s1.json`, and `final/final_s2.json` contain the preserved target-scene measurements. Keys have the form `scene|protocol|draw|training_seed|output_stage`. The accompanying checkpoints contain the pair-matcher state dictionaries for the three training seeds.

`meta/` contains the two complementary development splits and the no-augmentation / no-whitened-channel ablations. The latter retains scalar analytical features. The no-whitened-channel checkpoint therefore uses `PairMatcher(use_white=False)`.

The manuscript summary uses draws 0, 1, and 2 for P1, P2, and P4, and draw 0 for P0 and P3. It first averages references within each training seed, then averages seeds, then averages scenes equally. Some original records contain additional diagnostic draws; these are not included in that common-reference summary. `tools/summarize_results.py` applies the stated selection.

`time_s` is a cumulative experimental timing record that includes additional stages and metric calculations. It is not an isolated inference-latency measurement.

Map arrays, preprocessing caches, and local implementations of third-party comparison methods are omitted from this release. The recent-method comparisons in the manuscript use attributed literature reports.
