# Open Food Facts HSR Prediction Public Release

This folder documents the compact public output for the Open Food Facts HSR
prediction experiment. The main CSV.GZ may be tracked in this folder or attached
as a GitHub release asset, depending on repository-size policy. The full
internal prediction files include product names, brands, categories, and
nutrient fields; those fields are intentionally omitted from this public
release. Users can join by `code` against the official Open Food Facts export.

## Main File

- `openfoodfacts_hsr_assignments_public_seed42.csv.gz`

Rows: `4,532,767`
SHA256: `0879fec90a14f6dee0ee8ce89a3ae0048e2abb1882d7c58001c373f4f357b194`

The assigned HSR columns are model-derived pseudo-labels, not official Open Food
Facts labels and not direct HSR-calculator outputs.

## Columns

- `off_source_row_id`: row index from the OFF export used in the assignment run.
- `code`: Open Food Facts product code/barcode.
- `assigned_hsr_our_data_context`: HSR assigned using the private labelled
  context data.
- `assigned_hsr_our_data_plus_openhsr_context`: HSR assigned using private
  labelled context data plus OpenHSR.
- `mapped_nutrient_count_*`: number of mapped nutrient fields available to the
  upstream prediction model.
- `assigned_hsr_context_abs_diff`: absolute difference between the two context
  assignments.
- `assigned_hsr_context_same`: whether the two context assignments are exactly
  equal.

## Summary

- Context exact agreement: `0.446`
- Mean absolute context difference: `0.327`
- Missing product code rows: `0`

## Licence And Attribution Notes

Open Food Facts product data should be accessed from Open Food Facts and reused
under its database licence and attribution requirements. This release provides
model-derived HSR pseudo-labels keyed by OFF product code rather than a full
redistribution of product metadata or nutrition data.
