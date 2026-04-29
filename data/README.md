# Data

Raw recipe data is not redistributed in this portfolio repository.

To reproduce the workflow with your own data, place files under:

```text
data/raw/
  train/
  dev/
  test/
```

The parser expects recipe-style text files containing records with `Title:`, `ingredients:`, recipe instruction lines, and `END RECIPE`. Files ending in `.mmf`, `.MMF`, `.txt`, or `.TXT` are supported.
