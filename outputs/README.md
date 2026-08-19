# Synced experiment reports

`copy_helper.sh sync` copies only small, human-readable experiment reports
here: metrics, predictions, logs, and PNGs. It intentionally excludes model
weights, checkpoints, optimizer state, and model caches.

This directory is versionable so a completed experiment can be inspected on a
different cluster node or in a fresh checkout. Before committing new reports,
check their size and review them for unintended sensitive content:

```bash
du -sh outputs
git status --short outputs
```
