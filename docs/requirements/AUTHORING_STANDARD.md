# Requirements document authoring standard

This standard applies to every requirements document in this directory — the component
documents that follow the eleven sections below, and the study documents that follow the
shorter template further down. These documents are implementation briefs, onboarding
guides, component contracts, test specifications, architectural decisions, and handoff
records. They are written
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

## Study document template

Some deliverables (D08, D09, D10) are research, not code: a comparison to run, a question to answer, a
report to write. Forcing the eleven sections onto them produces empty headings. A
*study* document uses these sections instead, in this order:

1. Summary — what is being found out, in a paragraph.
2. Question(s) — each one a sentence that a result can answer yes or no.
3. Protocol — data, split, methods, metrics, and what is held fixed.
4. Inputs — data and code, with paths and their state today.
5. Subtasks — a table: id, what, needs (subtasks or decisions), done when.
6. Deliverable — the document or figure that comes out, with its path.
7. Acceptance — how a reviewer tells the deliverable is honest and complete.
8. Dependencies — which other deliverable and which owner to ask, and any `ISS-xx` by id.
9. This week and backlog — which subtasks are in the sprint, which carry over.
10. Findings — empty at first; filled in as the work runs.

A subtask that is really code (a reader, a converter) is written inside the study document
in the component style: `FRxx` and `TESTxx` lines with observable completion. The first
line of every document says which template it uses.

## Writing so a newcomer can follow

Every finding about code or method is written in three steps: what the code does now,
what the paper or contract expects instead, and why the difference matters for the
result. If the behaviour is not obvious, give a tiny concrete example (three gauges on a
line, one neighbour each). Define a term the first time it is used, in the sentence
where it appears — a glossary link does not replace that. Do not use shorthand that
assumes the reader already knows the method ("z-score per node type", "round-robin",
"O(0.1)"); say what happens instead ("subtract the training average and divide by the
spread"; "hands the stations out one at a time to fold 0, 1, 2…"). Do not name a
conclusion in a heading-style label without the mechanism under it.

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
  implementation-specific tests. Every document's Testing or Acceptance section says
  which modules it touches and that the coverage floors in rule 10 of the README (80 %;
  90 % for modules that write shared files) hold for them.
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
