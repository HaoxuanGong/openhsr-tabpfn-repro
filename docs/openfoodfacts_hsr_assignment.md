# Open Food Facts HSR Assignment

This document describes the optional Open Food Facts (OFF) HSR pseudo-labelling workflow used in the HSR prediction manuscript.

## What the Workflow Does

The workflow assigns model-derived Health Star Rating (HSR) pseudo-labels to Open Food Facts products using TabPFN-3 conditioned on labelled context rows.

Two context settings were used in the manuscript analysis:

1. `our_data_context`: labelled proprietary HSR rows only.
2. `our_data_plus_openhsr_context`: labelled proprietary HSR rows plus all labelled OpenHSR rows.

The output column `assigned_hsr` is a model-derived pseudo-label. It is not an official HSR calculator output.

## Assignment Quality

`assignment_quality` is derived from two flags:

- `low_feature_count`: true when `mapped_nutrient_count < low_feature_count_threshold`.
- `seed_unstable`: true when multi-seed predictions disagree beyond the configured thresholds.

The default low-feature threshold is 4 mapped nutrition fields.

The quality labels are:

- `high`: not low-feature and not seed-unstable.
- `low_feature_count`: too few mapped nutrition fields, but no seed-instability flag.
- `seed_unstable`: enough mapped nutrition fields, but unstable across seeds.
- `low_feature_count_and_seed_unstable`: both flags are true.

For a single-seed run, `seed_unstable` is always false because no across-seed variation can be estimated. In that case, `assignment_quality` mainly reflects feature completeness.

For reporting, use both:

- the full OFF pseudo-labelled catalogue, and
- a higher-confidence subset such as `assignment_quality == "high"` or `mapped_nutrient_count >= 6`.

## Publication and Licensing Notes

Code can be published in this repository, but do not commit:

- proprietary labelled context data,
- Open Food Facts full export files,
- generated OFF prediction CSV/CSV.GZ files,
- TabPFN model weights or cache directories,
- `TABPFN_TOKEN` or other credentials.

Open Food Facts data is distributed under the Open Database License (ODbL). If you publish a database or table that includes OFF product fields, provide Open Food Facts attribution and comply with ODbL share-alike requirements. A safe attribution statement is:

> Contains information from Open Food Facts, available under the Open Database License (ODbL): https://world.openfoodfacts.org/data

TabPFN model weights and inference are governed by Prior Labs licensing. Do not redistribute model weights or tokens. Users must obtain their own TabPFN access and comply with the relevant Prior Labs license.

## Recommended Paper Wording

> We assigned Open Food Facts products model-derived HSR pseudo-labels using TabPFN-3 conditioned on labelled context rows. These assignments are not official HSR calculator outputs. Products with sparse mapped nutrition information were flagged using `assignment_quality` and `mapped_nutrient_count`.
