# Grounding validation

The coverage engine reports each simulator->oracle mapping with a grounding
class: `direct`, `derived`, `proxy`, or `unmapped`. `proxy` means the simulator
fact is an engineering approximation of the oracle predicate (e.g. `jaywalking`,
`waiting`), not a measured equivalence. Until those are checked by a human, the
semantic coverage numbers rest on unvalidated labels.

`research/scripts/grounding_validation.py` makes the review explicit and
auditable:

```bash
# emit one row per crosswalk entry (sim predicate, oracle predicate, grounding,
# evidence string, blank label/notes columns)
python research/scripts/grounding_validation.py sheet --out research/logs/grounding/groundtruth.csv

# a human fills `label` (yes/no/unsure) and may correct `notes`; then:
python research/scripts/grounding_validation.py report --labels research/logs/grounding/groundtruth.csv
```

`report` prints the grounding breakdown, the labelled fraction, agreement over
labelled rows, and lists any `proxy` mapping that is still unlabelled. It exits
non-zero in that case (override with `--allow-unvalidated`) so a campaign cannot
silently claim validated grounding.

This is the code side of the task; producing the labels themselves is manual
review work and is not automated. Publish the sheet, the reviewed labels, and
the resulting agreement rate alongside the coverage results.
