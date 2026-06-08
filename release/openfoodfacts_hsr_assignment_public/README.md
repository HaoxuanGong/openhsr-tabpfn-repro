# Open Food Facts HSR Prediction Public Release

This directory contains the compact Open Food Facts HSR prediction table and
metadata for the manuscript release. The table is keyed by Open Food Facts
product `code`, so product names, brands, categories, and nutrient values can be
joined from the official Open Food Facts export.

## Main File

```text
openfoodfacts_hsr_assignments_public.csv.gz
```

Rows: `4,532,767`  
SHA256: `0879fec90a14f6dee0ee8ce89a3ae0048e2abb1882d7c58001c373f4f357b194`

The HSR columns are model-derived predictions, not official Open Food Facts
labels and not direct HSR-calculator outputs.

## Columns

- `off_source_row_id`: row index from the OFF export used in the prediction run.
- `code`: Open Food Facts product code/barcode.
- `assigned_hsr_our_data_context`: predicted HSR using the manuscript labelled
  context data.
- `assigned_hsr_our_data_plus_openhsr_context`: predicted HSR using the
  manuscript labelled context data plus OpenHSR.
- `mapped_nutrient_count_*`: number of mapped nutrient fields available to the
  upstream prediction model.
- `assigned_hsr_context_abs_diff`: absolute difference between the two context
  predictions.
- `assigned_hsr_context_same`: whether the two context predictions are exactly
  equal.

## Summary

- Context exact agreement: `0.446`
- Mean absolute context difference: `0.327`
- Missing product code rows: `0`

## Licence And Attribution Notes

Open Food Facts product data should be accessed from Open Food Facts and reused
under its database licence and attribution requirements. This file provides HSR
predictions keyed by OFF product code; it is not a redistribution of the full
Open Food Facts product database.
