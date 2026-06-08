# Open Food Facts HSR Predictions

This note describes the Open Food Facts (OFF) HSR prediction output released
with the repository.

## Overview

The OFF workflow predicts Health Star Rating (HSR) values for products in the
Open Food Facts export using TabPFN-3 and labelled HSR context rows.

The manuscript uses two context settings:

1. `our_data_context`: labelled HSR rows from the manuscript dataset.
2. `our_data_plus_openhsr_context`: the same labelled rows plus all OpenHSR rows.

The released table is keyed by OFF product `code`. The HSR columns are model
predictions, not official Open Food Facts labels and not direct outputs from the
HSR calculator.

## Public File

The compact file is stored at:

```text
release/openfoodfacts_hsr_assignment_public/openfoodfacts_hsr_assignments_public.csv.gz
```

It contains:

- `off_source_row_id`
- `code`
- `assigned_hsr_our_data_context`
- `mapped_nutrient_count_our_data_context`
- `assigned_hsr_our_data_plus_openhsr_context`
- `mapped_nutrient_count_our_data_plus_openhsr_context`
- `assigned_hsr_context_abs_diff`
- `assigned_hsr_context_same`

The table keeps only the fields needed to identify products, use the predictions,
and compare the two context settings. Product names, brands, categories, and
nutrition fields can be joined from the official Open Food Facts export.

## Data Use

Open Food Facts data is distributed under the Open Database License (ODbL). If a
new table or database redistributes OFF product fields, provide Open Food Facts
attribution and follow the ODbL share-alike requirements.

A suitable attribution statement is:

> Contains information from Open Food Facts, available under the Open Database
> License (ODbL): https://world.openfoodfacts.org/data

TabPFN model weights and inference are governed by Prior Labs licensing. Model
weights, cache directories, and access tokens should not be redistributed.

## Suggested Citation Wording

> We release a compact Open Food Facts table keyed by product code with
> model-derived HSR predictions from two TabPFN context settings: the labelled
> manuscript dataset alone and the same dataset augmented with OpenHSR. These
> values are predictions rather than official Open Food Facts labels or direct
> HSR-calculator outputs.
