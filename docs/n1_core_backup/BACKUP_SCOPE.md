# N1 Core Recovery Snapshot

Date: 2026-08-06

This branch is a recovery snapshot of the current N1-related implementation in
the DNN_Agent workspace. It is not the clean N1 publication repository and must
not be used as the source history for `git@github.com:ddriveb/N1.git`.

## Baseline

- Repository baseline: `0965cd4 chore: consolidate Strict v1.3 research workspace`
- Snapshot source: the current working tree under `/mnt/d/project/DNN_Agent`
- Selection authority: `docs/n1_export_audit/MIGRATION_MANIFEST.csv`

## Included

- N1 core, environment, baseline, training, experiment, and test files selected
  by the migration manifest.
- The 19 temporary phase-1 dependency files needed to reconstruct the current
  import closure.
- Audit documents, small result summaries, and the nine unique deployment/test
  weight files referenced by the audit.
- `AGENTS.md`, which records the canonical project environment.

## Excluded

- Regenerable training datasets and large artifact trees.
- Virtual environments, Python caches, native kernels, profiles, and temporary
  diagnostics outside the locked manifest.
- Unrelated changes from the larger DNN_Agent working tree.

The formal N1 migration remains governed by `docs/n1_export_audit/MIGRATION_PLAN.md`.
