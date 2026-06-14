<!--
SYNC IMPACT REPORT
==================
Version change: (template, unversioned) -> 1.0.0
Bump rationale: Initial ratification. Template placeholders replaced with concrete
governing principles, so this is the first stable MAJOR version.

Modified principles:
  - [PRINCIPLE_1_NAME] -> I. Code Quality Standards
  - [PRINCIPLE_2_NAME] -> II. Test Discipline (NON-NEGOTIABLE)
  - [PRINCIPLE_3_NAME] -> III. Disciplined Version Control
  - [PRINCIPLE_4_NAME] -> (removed; folded into the three principles above)
  - [PRINCIPLE_5_NAME] -> (removed; folded into the three principles above)

Added sections:
  - Quality Gates (replaces [SECTION_2_NAME])
  - Development Workflow (replaces [SECTION_3_NAME])

Removed sections: none (template slots repurposed)

Templates requiring updates:
  - .specify/templates/plan-template.md          OK (Constitution Check gate is generic)
  - .specify/templates/spec-template.md          OK (no mandatory sections changed)
  - .specify/templates/tasks-template.md         OK (already includes test + per-task commit guidance)
  - .specify/templates/commands/*.md             N/A (directory not present)

Follow-up TODOs: none
-->
# OrganMasker Lite Constitution

## Core Principles

### I. Code Quality Standards

Code MUST be readable, self-consistent, and maintainable before it is considered done.

- Every change MUST pass the project's configured linter and formatter with no new warnings.
- Public functions, modules, and CLI commands MUST have clear names and documented intent;
  no dead code, commented-out blocks, or unexplained magic values are merged.
- Complexity MUST be justified: prefer the simplest solution that satisfies the requirement,
  and record any deliberate deviation in the plan's Complexity Tracking table.
- All code MUST be reviewed (self-review at minimum for solo work, peer review when available)
  against these standards before merge.

Rationale: Quality is cheapest to enforce continuously. Consistent, lint-clean, well-named
code keeps the cost of every future change low and makes review meaningful.

### II. Test Discipline (NON-NEGOTIABLE)

Every behavioral change MUST be backed by automated tests.

- New features and bug fixes MUST add or update tests that demonstrate the intended behavior;
  a bug fix MUST include a test that fails before the fix and passes after.
- Tests MUST be written so they can fail: write the test, observe it fail, then implement.
- The full test suite MUST pass before any commit is pushed or any task is marked complete.
- Tests MUST be deterministic and independent of execution order; flaky tests are treated as
  defects and fixed or removed, not ignored.

Rationale: Proper tests are the contract that lets code evolve safely. Requiring a failing
test first proves the test exercises the change rather than the implementation proving itself.

### III. Disciplined Version Control

Work MUST be recorded as small, coherent git commits with meaningful history.

- Each commit MUST represent one logical change and MUST leave the codebase in a working state
  (builds and passes tests).
- Commit messages MUST be written in the imperative mood with a concise subject line and, when
  the change is non-trivial, a body explaining the "why".
- Commits MUST NOT bundle unrelated changes; formatting-only and behavioral changes SHOULD be
  separated.
- Secrets, generated artifacts, and local environment files MUST NOT be committed.

Rationale: A clean, atomic history makes changes reviewable, reversible, and debuggable.
Good commits are the most durable documentation of why the code is the way it is.

## Quality Gates

The following gates MUST be satisfied before a change is merged:

- Linting and formatting pass with no new violations.
- The full automated test suite passes locally.
- New or changed behavior is covered by tests (Principle II).
- The change is captured in atomic, well-described commits (Principle III).

A change that cannot meet a gate MUST either be revised to comply or have an explicit, recorded
justification in the plan's Complexity Tracking section before proceeding.

## Development Workflow

- Specifications and plans drive implementation; the Constitution Check in the plan MUST be
  re-evaluated against these principles before Phase 0 and after Phase 1 design.
- Implement in the order: write failing tests, make them pass, refactor, then commit.
- Commit after each task or logical group of changes, keeping each commit green.
- Reviews (peer or self) MUST verify compliance with all three core principles; unjustified
  violations block merge.

## Governance

This Constitution supersedes conflicting practices for this project.

- Amendments MUST be proposed as a change to this file, including the rationale and the version
  bump, and MUST update any affected templates and guidance in the same change.
- Versioning follows semantic versioning: MAJOR for backward-incompatible governance or
  principle removals/redefinitions, MINOR for new principles or materially expanded guidance,
  PATCH for clarifications and non-semantic refinements.
- Compliance is reviewed at every code review and plan checkpoint. Any deviation MUST be
  justified in writing (Complexity Tracking) or remediated before merge.
- For runtime, technology, and structure guidance, refer to the active plan and CLAUDE.md.

**Version**: 1.0.0 | **Ratified**: 2026-06-14 | **Last Amended**: 2026-06-14
