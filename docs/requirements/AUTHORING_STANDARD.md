# Requirements document authoring standard

This standard applies to every component requirements document in this directory.
These documents are implementation briefs, onboarding guides, component contracts,
test specifications, architectural decisions, and handoff records. They are written
for a capable third-year computer-science or data-science student who is new to the
domain and inherited repositories.

The owner should finish reading one document knowing what the component does, why it
exists, what it receives, what to build, which decisions are already made, what it must
hand downstream, and how to prove it works.

## Required structure

Use these top-level sections unless a section genuinely does not apply:

1. Summary
2. System Position
3. Inputs
4. Functionality Requirements
5. Non-negotiable Rules
6. Outputs
7. Testing
8. Documentation Requirements
9. Dependencies / Blockers
10. Current Technical Findings
11. Resources / Glossary

Include one short worked end-to-end example under **System Position**. Do not add an
“Implementation Freedom” section. Normal engineering judgement applies only to details
the requirements deliberately leave unspecified.

## Authoring rules

- Bias toward decisions. If a choice affects integration, reproducibility, scientific
  validity, maintainability, or team coordination, prescribe it.
- Explain a non-obvious prescribed decision with a **Why** subsection containing no
  more than two or three concise bullets.
- State who creates and maintains every settings file. Separate component-owned settings
  from external services and files received from earlier components.
- Use `FRxx`, `OUTxx`, `TESTxx`, and `DOCSxx` identifiers. Requirements should state
  observable completion criteria.
- Make outputs concrete: path convention, format, minimum fields, version identifier,
  and the component or teammate that reads them next.
- Start testing with component-level acceptance tests, then add source- or
  implementation-specific tests.
- Separate decisions from investigation. Requirements state what the team has decided;
  **Current Technical Findings** records evidence that may later change.
- Mark unresolved facts as `BLOCKED`, `TO VALIDATE`, or `UNKNOWN`. Do not present a
  candidate product or inferred fact as settled.
- Define unfamiliar acronyms, product names, file formats, and unavoidable technical
  terms at first use. A glossary link does not replace an explanation in the document.
- Apply the **pictureable noun test**: every important noun must refer to something the
  owner can mentally picture. Prefer `raw gauge file` to `source object`, `raw data from
  Component 01` to `upstream landing`, `input check` to `preflight`, and `write` to
  `materialise` or `emit`.
- Do not use data-platform shorthand such as `artifact`, `landing`, `object`, `schema`,
  `manifest`, or `content-addressed` merely because it sounds precise. If an exact
  filename such as `manifest.json` is required, give it a plain-language label and
  explain why it exists where it first appears.
- Use plain English and enough detail to eliminate important rediscovery work without
  adding a separate explanatory document.

## Project decisions that every document must respect

- The baseline study period is **2021-01-01 through 2022-12-31**. More years require a
  separately approved extension.
- The standard (canonical) rainfall dataset uses millimetres and timezone-aware UTC.
- The standard station key is the Australian Bureau of Meteorology (BoM) internal
  `station_id`, not the non-unique public station number.
- All randomness comes from `config/seeds.yaml` through the shared seed loader.
- Every method predicts the exact held-out records in a versioned split file.
- Every generated file set records its data version, settings, code revision, relevant
  seeds, and the exact input files/versions from which it was produced.
- Inherited scientific behaviour remains reproducible when corrected behaviour is
  introduced. Both must be explicitly labelled.
- Missing or excluded data is reported with a cause. Nothing is silently dropped.

## Review checklist

Before a requirements change is accepted, a reviewer should be able to answer “yes” to
all of the following:

- Can a new owner understand this component without reading the inherited code first?
- Are responsibility boundaries and the people/components that read each output explicit?
- Are important choices made rather than delegated as open-ended design exercises?
- Are all outputs sufficiently specified for the next component to read without private
  assumptions?
- Do the tests prove component behaviour, source traceability, failure handling, and handoff?
- Are unavoidable formats and technical terms explained at first use in this document?
- Are uncertainties visibly separated from requirements?
- Does the worked example show the component in context?
