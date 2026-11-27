# Predicate optimization in the Go expression pipeline

Status: implemented behind a temporary, default-off switch; scoped correctness verification is available, broader rollout is not approved. This document describes the implementation at `8d57a8df87` and the subsequent readiness audit. No legacy rewrite rule is retired.

## Problem and selected approach

Milvus currently parses and rewrites predicates in Go and exports `planpb.Expr` for native execution. Integrating a typed, reusable rewriting framework enables contextual and array simplification without moving schema interpretation or request policy into that framework.

The selected approach translates the existing Go rewrite result into `predicate-optimization` expressions, runs a fixed pass sequence and translates back. Alternatives are to port the framework directly to protobuf expressions, or replace existing rewriting immediately. Translation retains the framework's typed immutable graph and reusable rules, at the cost of extra planning allocations and time. Keeping existing rewriting active permits incremental migration with a known executable fallback. Smaller expression graphs alone do not establish lower query latency.

The dependency is pinned in [go.mod](../../../go.mod) to `github.com/lie-yan/predicate-optimization v0.0.0-20261126142937-e6870174d759`. No protobuf or native executor protocol extension is required.

## Pipeline and activation

The shared [parser](../../../internal/parser/planparserv2/plan_parser_v2.go) performs:

1. Parse and bind template values, including request-scoped membership preflight.
2. Notify the existing parsed-expression observer.
3. Run existing Milvus Go rewriting and placement validation.
4. When enabled and the source text is nonblank, assemble a schema view and run import → rewrite → export → candidate validation.
5. Return either the complete accepted candidate or the unchanged legacy expression. Ordinary caller plan assembly and final serialized-size checks continue afterward.

`proxy.enablePredicateOptimization` is refreshable and defaults to `false` in both paramtable and YAML. Each predicate parse captures it once. A main predicate and its scorers can capture different settings if configuration changes between those parses. This temporary switch is for rollout/comparison; it is not intended as permanent configuration after integration matures.

Query/count, search/hybrid, boost scorer filters and direct retrieve/search parser APIs use this hook. Delete and C-wrapper APIs also reach it when they call these constructors; there is no Role-based exclusion. Prebuilt query/requery plans, identifier-only and template-only APIs, blank predicates and disabled parses bypass the adapter. Parse/fill/initial validation failures return before attempting optimization.

## Types, values and translation

The adapter binds references to Milvus field IDs and the schema helper already used for parsing. The view checks top-level membership and schema metadata, including nullability. Nested paths and element scopes require separate contracts; their existence in the helper's flattened lookup does not admit them.

Current mappings cover supported combinations of:

| Source | Framework representation / export |
| --- | --- |
| Constants, AND/OR/NOT | Typed literal/logical expressions; n-ary AND/OR exports to an order-preserving balanced binary tree. |
| Unary comparisons and field comparisons | Resolved comparison calls with exact supported operand types. Export requires operands representable by Milvus comparison nodes. |
| Binary ranges | Two comparisons joined by AND; export need not reconstruct the original `BinaryRangeExpr`. |
| Scalar IN | Typed membership representation; executable export can normalize candidate order and duplicates. |
| Physical-array contains / contains-any | Typed built-in calls; `JsonContainsExpr` is admitted for physical arrays, not arbitrary JSON values. |
| Admitted NULL checks | Opaque expressions with known Boolean result type, preserving the native payload. |

Bool, signed integer widths, Float/Double and bounded VARCHAR are supported where the relevant mapping and function declaration admit them. Physical arrays admit supported Bool/numeric non-null elements and supported container nullability. This is not blanket support for every operator/type combination. Nullability is part of effective types; values and function signatures must preserve three-valued semantics. Implicit widening, general casts, arithmetic, arbitrary function calls, JSON/dynamic paths, element scopes and Bloom/Roaring nodes are not generally translated.

Unsupported descendants cause retention of the whole predicate. A result of unknown type cannot be safely invented or hidden in an opaque node. A future enclosing region with independently known type may provide a safe boundary, but this is not implemented general fallback coverage.

Export targets semantic equivalence and executable structure, not protobuf identity. Resolved template flags/names are discarded; ranges may become conjunctions; graph transformations may reorder or regroup supported forms. Bounds, field bindings and executable payloads remain required. No structural occurrence table is required to reconstruct the original syntax.

Sources: [translation dispatch](../../../internal/parser/planparserv2/predicateadapter/translation/import.go), [schema view](../../../internal/parser/planparserv2/predicateadapter/schema/schema_view.go), [export](../../../internal/parser/planparserv2/predicateadapter/translation/export.go).

## Runtime, ownership and cancellation

A process-wide immutable runtime supplies shared types, Value factories, function catalog/context and exact function resolution. It publishes once; initialization failure is cached until restart. Function declarations resolve from argument types. This pass sequence uses metadata and does not require function evaluation/constant folding.

Each attempt has its own translator, imported graph and field bindings. The caller's schema and legacy source are borrowed immutably. Acceptance transfers a complete executable candidate only after the real placement validator accepts it; no partial candidate is published. Immutable metadata/opaque payload sharing does not promise a deep copy of every descendant. There is no shipped optimization-result cache.

The production boundary uses `context.Background()` deliberately: translation runs synchronously to completion rather than threading request cancellation through all rules. The surrounding request machinery still handles request cancellation. The generic orchestration API recognizes cancellation/deadline failures as aborts; it does not silently retain after cancellation.

Sources: [integration](../../../internal/parser/planparserv2/predicateadapter/integration/integration.go), [orchestration](../../../internal/parser/planparserv2/predicateadapter/orchestration/roundtrip.go), [runtime publication](../../../internal/parser/planparserv2/predicateadapter/runtime/runtime_publication.go).

## Passes and outcomes

The [fixed sequence](../../../internal/parser/planparserv2/predicateadapter/orchestration/rewrite.go) is scoped contextual rewriting, Boolean cleanup, logical flattening, array membership normalization and array exclusion grouping. `RoundTrip` remains an explicit zero-pass baseline; production uses `Optimize`.

| Outcome | Boundary behavior |
| --- | --- |
| Translated | Use the accepted candidate. |
| Runtime unavailable | Retain the legacy source. |
| Unsupported import/export coverage | Retain the legacy source. |
| Classified translation defect, rule failure or candidate rejection with safe original | Retain the legacy source. |
| Unsafe source, invalid setup, unclassified unsafe failure or cancellation | Return an adapter-marked error; do not expose a candidate. |

Stages are Setup, Import, Rewrite, Export and Validate, plus Unknown for the zero/unclassified state. Diagnostic outcomes are not new public RPC error codes. Proxy error wrapping preserves adapter/system provenance and causes; ordinary parser errors keep their existing projection. No new error taxonomy, metric label or wire mapping is introduced by this design document.

## Existing request guards

Membership blob occurrence bytes and estimated Roaring decoded bytes are charged before both rewrite stages. Repeated blob references are charged per occurrence even when structural validation is cached or rewriting later removes the predicate. Search main/scorer filters share one budget; hybrid sub-requests share it across the request.

The final guard measures the actual complete membership-bearing `PlanNode` before marshaling, including scorers and caller-populated fields. Hybrid sub-plans share an accumulated size. Plans without Bloom/Roaring membership consume none of this particular budget. This is not a general expression-size cap. Over-limit failures remain input-classified code 1102 and non-retriable.

## Verification and limitations

Tests require actual translated/retained outcomes where result equality could conceal fallback. They cover immutable-source ownership, unsupported expressions, runtime failure, export defects, candidate rejection, cancellation and errors at query/search/scorer boundaries. Shared rewrite-failure tests verify rule errors and cancellation independently. Native touchstone and caller parity cover query/count/search/hybrid/scorers with flushed scan/INVERTED configurations; the earlier native snapshot is not proof of every later test/documentation revision.

The readiness audit corrects three parser assertions that required historical binary-range syntax. Expected bounds and inclusivity now explicitly describe the executable comparison pair; the tests are not removed. Independent translator conformance covers numeric widths, nullability, endpoint inclusivity and ownership. Arbitrary shape changes are not exempted from semantic or native performance verification.

Release benchmarks on Apple M5 measure the touchstone adapter at approximately 77 µs, with ~102 KB allocated per attempt. Its parser-built serialized plan shrinks by roughly 62%; synthetic stable predicates incur extra planning cost without shrinking. These are serial local planning measurements, not query-latency speedups or retained/peak-memory measurements. A dedicated ordinary-binary native count-query run on 6,144 rows observed ~17% lower touchstone median latency without scalar indexes, no clear indexed-touchstone benefit, and ~20% higher median latency for unchanged 128-branch predicates with/without INVERTED. Each setting has 50 measured samples in off/on/on/off blocks after warmup. These workload-dependent results do not justify blanket enablement.

Unrestricted full-suite success is not claimed. Exhaustive native coverage of all admitted types, growing segments, index families, delete behavior and C-wrapper error propagation remains outside the scoped evidence. No production default change follows from passing selected tests.

## Rollout and migration

Keep the default off and retain existing Go/native processing. Before enabling more broadly, reconcile the native latency results and exact review revision, investigate feature-specific failures, and document remaining verification exclusions. A feature PR must link this design document and its associated issue; no PR or issue is created by this document.

Further migration proceeds one legacy rule family at a time: inventory its exact domain and normalization duties; complete provider/export prerequisites; verify the new rule with the old rule disabled in the comparison harness; test composition and downstream performance; only then propose explicit retirement. Mandatory normalization must remain owned throughout. Candidate work includes Boolean/logical structure, scalar IN/ranges, arrays, text-specific rules and common-factor extraction. The existing five passes do not prove those legacy responsibilities can all be removed.
