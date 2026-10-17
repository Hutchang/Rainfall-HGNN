# Historical baseline evidence

Files in this directory are retained for provenance. They are not current implementation
requirements and should not be assigned directly to a component owner.

- [BASELINE_RECONCILIATION_REQUIREMENTS.md](BASELINE_RECONCILIATION_REQUIREMENTS.md)
  is the superseded reconciliation brief that preceded the component requirements.
- [AUDIT_NIRAMAY.md](baseline-reconciliation/AUDIT_NIRAMAY.md) records findings about
  the inherited Greater Sydney gauge data and graph path.
- [AUDIT_JYLOW.md](baseline-reconciliation/AUDIT_JYLOW.md) records findings about the
  second inherited implementation, including systems now outside project scope.
- [DATA_INVENTORY.md](baseline-reconciliation/DATA_INVENTORY.md),
  [SYSTEM_MAP.md](baseline-reconciliation/SYSTEM_MAP.md),
  [STATUS.md](baseline-reconciliation/STATUS.md), and
  [HANDOFF.md](baseline-reconciliation/HANDOFF.md) preserve the supporting investigation.
- [run_niramay_smoketest_native.log](baseline-reconciliation/run_niramay_smoketest_native.log)
  is the corresponding historical smoke-run record.
- [spatial-cv-and-regional-partitioning.md](spatial-cv-and-regional-partitioning.md) is the
  12 September design discussion of spatial leakage in the k-fold split and of one model
  per region. Nothing in it was implemented; ISS-10 tracks the split question.

[sprint1/](sprint1/) holds the closed sprint: the five component contracts that Sprint 1
was built to, [HANDOFF_HISTORY.md](sprint1/HANDOFF_HISTORY.md) recording its state and
resolved defects, and [COMPONENT_04_HANDOVER_NOTES.md](sprint1/COMPONENT_04_HANDOVER_NOTES.md),
whose fixes have all landed. The contracts still describe what each component implements
today, but they are reference, not current assignments.

Archived descriptions of Singapore/NEA or commercial microwave link behaviour are
historical evidence only. They do not override the Australian-only active requirements.

Current implementation work starts at [../CURRENT_STATE.md](../CURRENT_STATE.md) and
[../requirements/README.md](../requirements/README.md).
