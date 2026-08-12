# CLAUDE.md

## Core Engineering Rules

- **No backwards compatibility.** Never add compatibility shims, aliases, redirects, deprecated paths, migration wrappers, duplicate APIs, or fallback behavior. Update callers and delete obsolete code.
- **Reuse before creating.** Before adding any class, function, module, constant, enum, model, helper, or test utility, search the repository for an existing implementation to reuse, extend, merge, move, or refactor.
- **Refactor continuously.** Every change must improve or preserve architecture and code quality. Remove duplication, dead code, stale branches, redundant abstractions, obsolete tests, unused files, and unnecessary boilerplate encountered in the touched area.
- Prefer the **smallest coherent design**, not the smallest patch. Fix root causes rather than layering workarounds.

## Type & Domain Modeling

- **No primitive obsession.** Do not pass raw strings, integers, floats, booleans, tuples, or loosely structured containers when the value represents a domain concept.
- Use **Enums** for finite states, modes, strategies, identifiers, categories, policies, statuses, and options.
- Use **dataclasses** or other explicit typed models for structured data, configuration, records, inputs, outputs, state, and results.
- **No dictionaries as application contracts or state containers.** Do not use `dict`/mapping-shaped plumbing between modules. Convert external dictionary-shaped data at the boundary into typed models immediately.
- **No `Any`.** Use precise types, generics, protocols, unions, enums, and typed models.
- Avoid raw string identifiers and magic literals. Centralize meaningful constants and give them descriptive names.
- Public APIs and cross-module boundaries must be strongly typed and semantically explicit.
- Prefer immutable/value-style domain objects where mutation is unnecessary.
- Validate invariants at construction or at the closest responsible boundary.

## Architecture

- Keep responsibilities cohesive and dependencies one-directional.
- Do not create thin wrappers, pass-through modules, forwarding classes, or abstraction layers without real semantic value.
- Do not duplicate concepts under different names.
- Merge overlapping implementations instead of preserving parallel versions.
- Keep orchestration thin; domain/capability code must own actual behavior.
- Avoid circular dependencies and hidden global state.
- Keep I/O, framework integration, serialization, and third-party formats at explicit boundaries.
- Use descriptive names. Never introduce opaque aliases, numbered regimes, unexplained abbreviations, or temporary-looking production names.

## Implementation Discipline

Before writing new code:

1. Search globally for the concept, behavior, symbol, and nearby equivalents.
2. Inspect existing call sites, tests, models, enums, constants, and utilities.
3. Reuse or refactor existing code whenever possible.
4. Create something new only when no suitable abstraction exists.

When changing behavior:

- Update the canonical implementation and every affected caller.
- Delete the old implementation instead of preserving compatibility.
- Delete or rewrite stale tests rather than keeping tests for obsolete behavior.
- Remove code that is reachable only from tests unless it has a legitimate production purpose.
- Do not leave TODO implementations, placeholder branches, silent fallbacks, or speculative abstractions.
- Do not add comments that merely narrate obvious code or sound AI-generated. Comments must explain non-obvious intent, invariant, or scientific/technical rationale.

## Quality Gates

All changed code must pass:

- **Pylance** diagnostics in the IDE/workspace.
- **Pyright** with no unresolved type errors.
- **Ruff** linting.
- **Ruff formatting** / repository formatting rules.
- Relevant tests plus the full test suite when feasible.

Fix pre-existing issues in touched areas rather than working around them.

## Tests

- Tests must validate behavior, invariants, edge cases, failure modes, determinism, and integration paths—not implementation trivia.
- Prefer deterministic tests with explicit seeds where randomness exists.
- Run tests **in parallel** whenever the test framework supports it (for pytest, use `pytest-xdist`, e.g. `pytest -n auto`).
- Do not weaken, skip, xfail, or delete a valid test merely to make a change pass.
- If behavior intentionally changes, rewrite the test to assert the new canonical behavior and remove obsolete expectations.
- Keep fixtures and test helpers typed, reusable, and non-duplicative.
- Production functionality must not exist only to satisfy tests.

## Error Handling & Validation

- Fail fast on invalid states, unsupported combinations, missing required data, and violated invariants.
- Never silently coerce scientifically or semantically distinct values.
- Never swallow exceptions without a deliberate, typed recovery policy.
- Unsupported behavior should be rejected explicitly rather than approximated through fallback logic.
- Boundary validation must be explicit and deterministic.

## Configuration & Constants

- Prefer typed configuration objects, enums, dataclasses, and named constants.
- No magic numbers or magic strings in business/scientific logic.
- A value with semantic meaning must have one canonical definition.
- Do not duplicate configuration values across modules.
- Validate configuration combinations before execution.

## Performance & Maintainability

- Prefer efficient library/vectorized operations where they improve clarity and performance.
- Avoid unnecessary copies, repeated parsing, repeated I/O, repeated computation, and needless object churn.
- Optimize only with semantic correctness preserved, but proactively remove obvious inefficiencies.
- Keep modules and APIs small, cohesive, and discoverable.
- When a simpler architecture can replace accumulated boilerplate, refactor toward it.

## Repository Hygiene

- Do not leave dead files, stale exports, unused imports, commented-out code, backup files, temporary artifacts, or abandoned implementations.
- Do not create duplicate source trees or legacy namespaces.
- Keep package exports intentional and minimal.
- Rename misleading symbols instead of documenting around bad names.
- Keep generated/runtime artifacts out of source unless explicitly required.

## Completion Standard

A task is not complete until:

1. Existing code was searched for reuse opportunities.
2. The canonical implementation is clear and non-duplicated.
3. Obsolete code and compatibility paths are removed.
4. Types and domain models are explicit; no `Any` or dictionary-shaped application contracts remain in touched code.
5. Pylance/Pyright/Ruff/formatting are clean.
6. Tests pass in parallel.
7. The touched area has been refactored enough that no obvious duplication, dead code, primitive leakage, or unnecessary abstraction remains.
