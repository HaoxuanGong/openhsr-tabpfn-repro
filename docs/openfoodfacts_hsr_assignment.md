# Open Food Facts HSR Prediction Release

This document describes the optional Open Food Facts (OFF) HSR pseudo-labelling
workflow used in the HSR prediction manuscript.

## What the Workflow Does

The workflow assigns model-derived Health Star Rating (HSR) pseudo-labels to
Open Food Facts products using TabPFN-3 conditioned on labelled context rows.

Two context settings were used in the manuscript analysis:

1. `our_data_context`: labelled proprietary HSR rows only.
2. `our_data_plus_openhsr_context`: labelled proprietary HSR rows plus all
   labelled OpenHSR rows.

The public release is keyed by OFF product `code`. The assigned HSR columns are
model-derived pseudo-labels, not official HSR calculator outputs.

## Public Release Fields

The compact public file contains:

- `off_source_row_id`
- `code`
- `assigned_hsr_our_data_context`
- `mapped_nutrient_count_our_data_context`
- `assigned_hsr_our_data_plus_openhsr_context`
- `mapped_nutrient_count_our_data_plus_openhsr_context`
- `assigned_hsr_context_abs_diff`
- `assigned_hsr_context_same`

The public file intentionally omits OFF product names, brands, categories,
nutrition fields, private context rows, model weights, and authentication
tokens. Readers can join by `code` against the official Open Food Facts export
for downstream inspection.

## Publication and Licensing Notes

The compact public file under `release/openfoodfacts_hsr_assignment_public/` can
be published in this repository. Do not commit:

- proprietary labelled context data,
- full Open Food Facts export files,
- internal OFF prediction files that include product metadata or nutrition
  fields,
- TabPFN model weights or cache directories,
- `TABPFN_TOKEN` or other credentials.

Open Food Facts data is distributed under the Open Database License (ODbL). If
you publish a database or table that includes OFF product fields, provide Open
Food Facts attribution and comply with ODbL share-alike requirements. A safe
attribution statement is:

> Contains information from Open Food Facts, available under the Open Database
> License (ODbL): https://world.openfoodfacts.org/data

TabPFN model weights and inference are governed by Prior Labs licensing. Do not
redistribute model weights or tokens. Users must obtain their own TabPFN access
and comply with the relevant Prior Labs license.

## Recommended Paper Wording

> We released a compact Open Food Facts table keyed by product code with
> model-derived HSR pseudo-labels from two TabPFN context settings: the private
> labelled dataset alone and the private labelled dataset plus OpenHSR. These
> assigned values are not official HSR calculator outputs.
