# Predicate adapter package setup

This package establishes the Milvus-owned boundary for paired expression translation. The `values` subpackage exposes the bundled stateless ValueFactory as `values.Factory` through its public interface and supplies shared immutable NonNull/MayNull descriptors for Bool, Int8/16/32/64 and Float32/64. Factory capability remains broader than translator admission. Paired opaque preservation, native logical forms, unary/field comparisons, binary ranges and scalar membership are implemented for the documented D2 domains; standalone orchestration is implemented in `orchestration/`, while production wiring remains pending.

## Dependency resolution

The Milvus root `go.mod` requires `github.com/lie-yan/predicate-optimization` at the pseudo-version corresponding to submodule commit `198aa79c148c10b92165fe59f6a57b523ee0272e` and replaces it with `./third-party/predicate-optimization`. No dependency is added to `pkg/go.mod`.

Initialize the source dependency in a fresh checkout:

```sh
git submodule update --init third-party/predicate-optimization
```

Local builds use that checkout, including any uncommitted optimizer edits. Reproducible CI/release builds must initialize the submodule at the commit recorded by the Milvus checkout and use a clean optimizer tree. The local replacement means Go does not verify the selected source against the require version: update the requirement and Git submodule pointer together when advancing the dependency. Do not use a floating submodule branch for release builds. Source archives/build uploads must include this submodule or initialize it before invoking Go; no remote-only release build is established by this setup.

No optimizer go.sum entry is needed for the local replacement. Avoid a repository-wide go mod tidy merely to add this dependency. There is no go.work requirement or absolute developer path.

## Package direction and placement seam

The parent `planparserv2` package and audited proxy callers may import this package later. This package must not import either of them. Future code may use planpb, schema helpers and optimizer public interfaces/selected implementations.

The parser will pass its existing read-only placement validator to the future synchronous round-trip operation. That callback runs on a complete private candidate after adapter validation, before success is returned. The original placement check still runs before the hook. No callback type, RoundTrip stub, parser hook or caller enablement is added just to establish the package.

## Compilation check

From the Milvus root:

```sh
go test -tags dynamic,test -gcflags="all=-N -l" -run '^$' ./internal/parser/planparserv2/predicateadapter/...
```

This checks real optimizer imports and factory/interface compatibility without executing tests. It is a package compilation check, not interface-behavior conformance or a full Milvus binary build. The current dependency closure is pure Go and does not require a native SDK. If parent-parser code changes later, compile/test that package too using the required Milvus workflow.

## Package conformance

Run the scalar and column-identity tests from the Milvus root:

```sh
go test -mod=readonly -tags dynamic,test -gcflags="all=-N -l" -count=1 ./internal/parser/planparserv2/predicateadapter/...
```

The tests use public Value payload interfaces and an independent scalar representation. They cover exact Bool/Int64 payloads, symmetric public-interface equality, full-Type distinctions and nullability adaptation. Initial use is sequential; dedicated primitive-descriptor, factory-error, concurrent-construction, composite and wider-scalar tests are omitted at this stage. Broader bundled Value regression coverage runs in the optimizer module with `go test -mod=readonly -count=1 ./testbed/values/...`. These scalar tests do not establish protobuf admission/export, function comparison semantics or Milvus runtime parity; separate translation coverage is described below.

## Shared primitive Types

`runtime/types.go` names all 14 package-local primitive descriptors explicitly: `boolNonNullType`/`boolMayNullType`, `int8NonNullType`/`int8MayNullType` through Int64, and `float32NonNullType`/`float32MayNullType` through Float64. Treat them as immutable. Use these instances instead of repeatedly constructing equivalent primitive Types; parameterized Types remain constructed as needed.

The private `mustPrimitiveType` helper only unwraps constructors with fixed valid declaration arguments. It is not request validation. Descriptor availability alone does not grant admission. Native mapping domains differ by expression form and currently require nonnullable top-level fields; approved opaque null predicates can preserve nullable fields.

## Session tokens and column identity

`utils/session_token.go` supplies process-local, non-recycled session tokens. The `expressions` subpackage mirrors the testbed layout: `expressions/column_id.go` provides `ColumnID` and `NewColumnID(session uint64, fieldID int64)` for immutable column keys of the form `mvcol:v1:<session>:d:<field-id>`. Type and display name are separate from identity. Equality compares canonical keys through the public interface, including compatible external implementations.

Future session creation must allocate through `utils.NextSessionToken()`, which uses a single private allocator. Zero is invalid; exhaustion returns `(0, false)` permanently without wrapping. Atomic reservation protects the process-wide counter. Tokens must not be persisted or transmitted, and the allocator must not be reset or copied after use. Schema membership and live-session ownership checks belong to later registry work.

Three focused tests cover canonical-key equality across representations and identity separation, sequential token lifetime/separation, and exhaustion using an isolated allocator. These checks do not establish concurrent session orchestration or schema/registry behavior.

## Pinned schema view

`schema/schema_view.go` (`schema.NewView`) pairs the parsing SchemaHelper with top-level field pointers from the same published read-schema entry. The constructor checks pointer agreement with helper lookup and copies only the membership slice. The caller must supply complete top-level membership and keep field nodes immutable throughout the attempt; the helper alone cannot prove that a supplied list is top-level. No schema envelope or proxy cache is retained.

`View.CheckColumn` returns a borrowed field and `supported=true` for consistent top-level columns regardless of datatype or nullability. Bool/Int64 are the motivating translation examples, not a schema-validation whitelist. A successful schema check does not establish translator coverage. Excluded scopes and unknown ColumnInfo fields return a coverage miss. For document columns, missing field IDs or datatype/nullability/element-type/key/auto-ID disagreement produce internal-contract errors. The helper's ParameterInvalid lookup error is deliberately relabeled ServiceInternal at this post-parser boundary, with its cause retained. Caller-visible outcome plumbing is still pending.

Three schema tests cover column scope, datatype-independent metadata consistency and pinned membership across a simulated entry replacement. No production caller ownership, cache refresh or round-trip behavior is established by these fixtures.

## Field registry

`schema/field_registry.go` (`schema.NewFieldRegistry`) owns one session token and one field-ID map per registry. `FieldRegistry.BindColumn` checks the pinned schema, validates reference nullability and reuses an existing identity only when the full Type agrees. The translator supplies the correctly derived immutable Type; the registry does not implement datatype conversion or a Bool/Int64 whitelist.

`FieldRegistry.ResolveColumn` accepts canonical keys through the public ColumnID interface, verifies the session prefix, decodes the original Milvus field ID, and checks registration, exact key and full Type. It returns the checked schema/ColumnInfo association for reverse translation without registering anything or refreshing the schema. Returned `FieldBinding` values expose `ColumnID`, `Type`, `Field` and `ColumnInfo` accessors while keeping their fields private; their IDs/Types are immutable and protobuf metadata remains borrowed read-only. Export must copy metadata into fresh output, with occurrence-specific provenance supplied separately.

Three focused tests verify binding reuse, consistency failures and isolated registry resolution, including compatible external IDs, nullable/narrow fields and foreign/unregistered keys. Registration is sequential; later validation/export must use the registry read-only. No caching, row binding or production wiring is included.

## Function catalog

`functions/catalog.go` implements S05.1 registration and immutable catalog ownership for comparison, arithmetic and ordinary functions. The builder deduplicates qualified names with equal full signatures, rejects conflicting semantic metadata and is consumed by a successful Build. Metadata and Types are shared immutable objects; copied registration values cannot change catalog records. Three focused tests cover identity, conflict rejection and lifecycle/ownership.

`functions/handle.go` implements S05.2 private handles and Metadata. Handles retain the catalog and canonical entry index; independent wrappers compare by that identity. Metadata rejects foreign representations/snapshots and unknown entries with the framework error categories. Two focused tests cover identity, metadata ownership and failure behavior. `functions/lookup.go` implements S05.3 exact qualified-name lookup through a shared candidate scan and the framework binder. It returns one canonical handle or reports no-match/ambiguity without coercion or evaluation; binding contract defects retain their category. Four tests cover binding, unsupported calls, ambiguity and an injected adaptation defect. S05.4 adds built-in/operator lookup through the same resolver and classification through validated metadata. Three focused tests cover alias identity, family/arity checks, ambiguity and handle errors. `functions/comparisons.go` completes S05.5 with six Int64 comparisons and Bool equality/inequality as ordinary registrations, sharing supplied non-null Types and PropagateNull policies. Two tests verify the eight definitions and mixed-catalog binding. Contexts delegate inspection/lookup to the catalog; implementation setup, checked execution and the complete context are implemented through S06.3. Initial internals use a linear scan; structural indexes and scalability benchmarks are deferred. S06.4 verifies no-evaluation composition; S07 adds test-only comparison implementations; one-time publication is implemented in S08.2.

## Internal assertions

`internal/assertions` provides `NotNil(any)` for interfaces (including typed nils) and `NotNilPointer[T](*T)` for pointers without reflection. The `debug` build tag enables panic checks; default builds use no-op implementations. Arguments are still evaluated, and invalid inputs may still panic when subsequently dereferenced. These are programmer preconditions, not input validation.

Function handles, lookup and ColumnID use the shared guards. Checks returning setup/schema/registry errors remain active in both configurations, as do non-nil construction-state checks such as handle index bounds and empty identity keys. Run adapter regressions with both `-tags dynamic,test` and `-tags dynamic,test,debug`.

## Function context

`functions/context.go` supplies `NewFunctionContextBuilder`, `BindImplementation` and `Build` for a complete immutable `FunctionContext`. The builder consumes its setup state on Build, retains the supplied catalog and freezes context-specific FunctionImplementation bindings. Lookup/classification/metadata delegate to the catalog; Evaluate uses the checked engine in `functions/evaluation.go`. Omitted implementations do not affect lookup and return unavailable execution when dispatch is required.

Contexts may share catalog handles while retaining different implementations; calls must retain their original context. Production and executable harness assembly use separate catalogs. Concurrent evaluation requires thread-safe implementations with stable captures. `tests/no_evaluation_test.go` completes S06.4 with independent Evaluate/implementation guards across lookup, inspection, call construction and graph validation. S07's real comparisons are verified in `tests/comparison_harness_test.go`; S08.2 provides one-time runtime publication; actual import/export no-evaluation verification follows translator implementation.

## Executable comparison harness

`tests/comparison_harness_test.go` builds isolated catalogs/contexts with eight test-only FunctionImplementations. Three tests cover direct signed Int64 comparisons (including endpoints and exact large integers), all Boolean pairs, public payload wrappers, nullable results/NULL propagation and cross-catalog rejection during call construction. S06.2 retains the generic invalid-Type and execution-error tests. The harness establishes callback conformance, not actual Milvus parity or production evaluation enablement.

## Runtime construction

The `runtime` subpackage groups bundle construction, one-time publication, shared Type descriptors and their focused tests. Callers import it directly; the sibling `orchestration` package owns the synchronous operation.

`runtime/runtime.go` provides `NewRuntime` for a complete isolated empty-pipeline bundle containing the captured factory, shared primitive descriptors, matching immutable catalog/context and semantic/policy identity. `Types` returns descriptor associations by value. `ComparisonFunction` maps a Milvus comparison enum into catalog lookup; `ComparisonOperator` validates handle ownership before recovering the Milvus enum from metadata. Neither method translates protobuf nodes or proves export eligibility.

Construction validates unique operator resolution before returning a bundle and binds placeholders with no specified invocation behavior. Empty translation must never evaluate. S08.1 adds construction/ownership and setup-failure tests; S08.2 provides one-time publication; S08.3 verifies standalone foundation composition. Do not construct a runtime per session.

## Shared runtime initialization

`SharedRuntime()` lazily initializes one process-wide bundle through `runtime/runtime_publication.go`. Concurrent callers wait for complete construction and receive the same runtime pointer. A setup error is cached unchanged until process restart; no retry, replacement or reset API is provided. `NewRuntime()` remains isolated construction, not the production acquisition path. No parser caller is enabled yet.

Live version replacement and old/new-bundle verification are deferred. This API publishes immutable runtime data only; it does not authorize evaluation, optimizer passes or production translation. `tests/foundation_composition_test.go` verifies shared runtime identity, isolated registries, reference/call construction, operation recovery and zero evaluation with independent interface/dispatch instrumentation. D1 is complete for the initial lifecycle; D2 paired translation is complete for the matrix below. D3 standalone orchestration is complete; actual Milvus wiring remains pending.

## Translation foundation

`translation/` defines the runtime capability interface, private occurrence associations and local stage/failure provenance for D2-M01. It borrows immutable runtime/source metadata, copies ordered child associations and preserves underlying error causes. It does not import the parent adapter. M02.3 adds callable admission/import/export/validation stages for approved opaque roots; no production hook is enabled.

## Opaque preservation identities

`expressions/opaque_id.go` provides immutable attempt-scoped OpaqueID keys. The `translation/utils/opaque_registry.go` utility assigns ordered region identities and resolves them only when canonical key, full Type and expected source occurrence agree. Source protobuf metadata remains borrowed in the occurrence table; no duplicate source tree or semantic-deduplication map is introduced. M02.2 tests cover identity, association ownership and invalid lookup; M02.3 supplies admission and paired opaque translation.

## Paired opaque translation

`translation.New` captures the runtime capabilities, pinned schema view and a value snapshot of `runtime.PrimitiveTypes`. The snapshot retains shared descriptor pointers; setup checks every admitted NonNull numeric descriptor and all 38 comparison signatures. `Admit` selects complete routing before `Import` allocates an opaque registry and graph. `Export` checks the actual graph before reconstructing a fresh protobuf; `ValidateCandidate` independently inspects emitted content. Admitted/imported state belongs to its creating translator; each import has distinct opaque identities.

The initial O1 route preserves top-level scalar `IS NULL`/`IS NOT NULL` over Bool, Int8/16/32/64, Float, Double and VarChar, including nullable fields; its result is NonNull Bool. Unknown fields, unsupported scopes/domains and unknown-result calls remain coverage misses. Malformed known payloads and schema mismatches remain source defects. M02.4 composes this route with native TRUE/NOT/binary AND/OR. No evaluation, optimizer pass or production caller is enabled. Source/schema data must remain immutable throughout the attempt; output protobufs are independent deep copies.

Stage state and dispatch live in `admission.go`, `import.go`, `graph_validation.go`, `export.go` and `candidate_validation.go`; `mapping_opaque.go` keeps the paired opaque handlers together. Independent utilities live in `translation/utils/`, while occurrence state and stage failures remain in the translation package.

`translation/tests/mapping_opaque_test.go` covers the approved domains, failure categories, actual graph identity/Type checks, independent output and candidate validation, foreign ownership and cancellation. These tests establish the isolated preservation path, not Milvus execution parity or parser integration.

## Native logical composition

`translation/mapping_logical.go` pairs AlwaysTrueExpr with a NonNull Bool true literal, UnaryExpr(Not) with NOT, and BinaryExpr(LogicalAnd/LogicalOr) with exactly binary AND/OR. FALSE remains NOT TRUE. Admission records ordered child routes and requires every child to have an admitted NonNull Bool result; unsupported children propagate coverage and malformed known children remain defects even after an unsupported sibling.

Import follows routes per occurrence without simplification or evaluation. Graph validation checks operator, arity, Type and ordered child correspondence. Export reads actual graph operators/children and constructs fresh native messages, retaining each source occurrence's IsTemplate flag. Candidate validation independently checks emitted forms and children. Opaque regions stay terminal and are copied only after identity resolution.

`translation/tests/mapping_logical_test.go` covers mixed trees, shared/equal occurrences, distinct metadata, independent output, coverage/defect propagation, changed graphs/candidates, n-ary rejection and cancellation throughout traversal. This verifies isolated empty-pipeline translation, not parser integration or C++ runtime parity. N-ary export remains a pre-rewriter follow-up.

## Field and literal operand codecs

`translation/operands.go` supplies private admission, import, graph checks, export and candidate checks for embedded top-level NonNull Bool, Int8/16/32/64, Float32/64 and VarChar operands. Field admission reuses the pinned schema view; import/export use the owning field registry and verify the expected field role. Each operand retains its own ColumnInfo for fresh output instead of using the first registry binding's metadata.

Literal admission checks actual GenericValue oneof arms, preserving false, zero and the complete signed Int64 range. Export reads exact public scalar payloads after Type/null/content checks and allocates a fresh message and arm. Small operand associations borrow immutable source metadata; they do not introduce bare ColumnExpr/ValueExpr predicate routes.

Seven tests cover exact payloads, source limitations/defects, field ownership/roles, rebuilt public payload views, candidate mutations, independent output and a factory failure whose cause survives import. M04 connects these codecs to complete Bool/Int64 comparisons and lazily allocates each import's field registry.

The wider codecs preserve declared integer/float widths and VarChar schema byte bounds. Literal Types are supplied by the enclosing mapping rather than automatically copied from the field: an Int8 field can require an Int64 comparison literal, and a query string can exceed a field's storage bound. Out-of-range payloads, non-exact Float32 values and over-bound strings are coverage misses; no narrowing, rounding or truncation occurs. NaNs remain excluded because the factory canonicalizes their bits. Floating output checks preserve signed zero explicitly. Wider codec availability does not enable additional comparison functions or nullable/composite operand admission.

## Unary comparisons

`translation/mapping_unary_range.go` pairs NonNull Int64 equality/inequality/order comparisons and NonNull Bool equality/inequality with field-first comparison calls. Admission requires concrete matching literal payloads and empty ExtraValues. Import uses the pinned runtime/context and one lazily allocated field registry per import; embedded operand associations retain occurrence-specific metadata.

Graph checks require the retained context, canonical handle/operator, ordered field identity and exact literal content. Export reads actual operands and emits a fresh specialized UnaryRangeExpr, preserving IsTemplate and TemplateVariableName; candidate checks independently inspect the emitted message. Wider/nullable comparisons and Bool ordering remain coverage misses. Casts remain deferred.

Five tests cover all eight mappings and reconstructed equivalent calls, mixed logical/opaque composition and independent output, admission boundaries, changed graph/candidate rejection, and cancellation at every stage. A guarded context rejects any evaluation. These checks do not establish parser wiring or Milvus execution parity.

## Field comparisons

`translation/mapping_compare.go` pairs CompareExpr with ordered reference/reference calls for NonNull same-width Int8/16/32/64 and Float32/64 (six operators), plus Bool equality/inequality. `functions/numeric_comparisons.go` and runtime assembly add 30 numeric declarations to the original eight. Mixed Types, nullable operands, VARCHAR and Bool ordering remain coverage misses; M04's unary scope is unchanged.

Both comparison forms share lazy registry creation and actual context/handle/operator checks in `translation/comparison.go`. Export preserves both occurrence-specific ColumnInfo objects and IsTemplate in independently owned specialized output. Swapped or foreign references fail validation, and same-field floating comparisons are never folded. Three translation tests cover mappings, admission and mixed-form ownership/rejection; one declaration test covers binding. No evaluation, optimizer pass, parser wiring or downstream runtime parity is enabled by this step.

## Translation test layout

`translation/tests/` contains public-stage tests and their shared fixtures in `helpers_test.go`. Tests for private operand codecs, provenance and failure construction remain in `translation/`; standalone utility tests remain in `translation/utils/`. Run `go test` with `./internal/parser/planparserv2/predicateadapter/translation/...` to include all three packages, using the flags documented above.

Variant handlers and associated files follow `mapping_<variant>[_<role>].go`; tests append `_test.go`. For example, `mapping_opaque_associations_test.go` covers private occurrence associations. Shared facilities and independent utilities retain their descriptive names.

## Binary ranges

`translation/mapping_binary_range.go` pairs numeric BinaryRangeExpr with an ordered AND of lower and upper comparison calls over one registered reference. Admission requires exact NonNull same-Type numeric bounds; out-of-width, conversion-dependent, NaN, nullable and VARCHAR cases remain coverage misses. Missing/malformed operands and parser-invalid bound ordering remain source defects.

A retained range descriptor distinguishes generated comparison roles from an ordinary conjunction, even when graphs are equal or shared. Export reads actual checked payloads/operators, preserves inclusivity and separate template names, and constructs independently owned specialized output. Three mapping tests plus the shared comparison cancellation test cover these contracts. Production wiring and execution parity remain pending.

## Scalar membership

`translation/mapping_term.go` pairs homogeneous nonempty TermExpr with SpecialFormIn for NonNull Bool, integer widths, Float32/64 and bounded VARCHAR. Target and candidates retain identical full Types; no catalog registration is needed. Existing NOT handling supplies NOT IN. Empty/reverse/nullable membership, coercion-dependent candidates, NaNs and over-bound strings remain coverage misses.

Admission accepts unsorted and duplicate candidates and preserves their sequence without normalization. Import preserves target-first candidate order; graph and candidate checks enforce exact roles, Types, payloads and metadata. Export allocates fresh specialized messages and candidate slices. Three mapping tests cover domains, homogeneous candidate coverage, graph/candidate rejection and ownership; `tests/cancellation_test.go` covers cancellation across unary, range and term mappings. No evaluation or production wiring is enabled.

## Composed validation

M08 verifies every implemented mapping in one mixed-domain tree and checks source coverage versus candidate defects for unknown protobuf fields. `translation.New` validates all admitted numeric descriptor widths/nullability and all 38 comparison signatures before returning a translator. Public graph validation and export run framework validation before adapter correspondence checks; emitted-candidate validation remains independent.

`translation/tests/composition_test.go` covers native/opaque composition, final-region failures, participating protobuf message versions and incomplete wider runtime setup. D3 still owns placement validation, final acceptance and caller outcomes; local export success is not publication or downstream execution parity.

## D2 completion and D3 handoff

D2 M01–M09 are complete for the achieved mapping matrix. Whole-tree conformance tests in `translation/tests/conformance_test.go` verify independent candidate ownership with shared native graph nodes and occurrence-specific metadata, plus zero Evaluate/implementation calls through every translation stage. Production comparison bindings remain placeholders; successful translation does not imply executable folding support.

The detailed mapping limits, verification commands, stage APIs, lifetime requirements and remaining acceptance responsibilities are recorded in [M09 — Ownership and D3 handoff](../../../../third-party/predicate-optimization/private-docs/Working/MilvusIntegration/PhaseD-ImplementationAndConformance/D2-PairedTranslators/M09-OwnershipAndD3Handoff.md). D3 now composes candidate validation, trusted placement validation and final cancellation/acceptance into one synchronous operation. No parser hook or live-Milvus parity is claimed here.

## Standalone orchestration

`orchestration.ObtainRuntime` captures explicit opt-in and the shared runtime outcome. `orchestration.RoundTrip` takes that runtime provision via `AttemptInput.RuntimeProvision`, borrowed source/schema, request context and a trusted read-only placement callback. It returns a fresh accepted candidate, a candidate-free retention result, or an error recognized by `orchestration.IsAdapterAbort`. The parser/proxy integration is not enabled.

D3 conformance covers local stage failures, cancellation, successive-attempt output isolation and independent zero-evaluation/implementation counters with real import/export. A private per-call translator constructor supports fault tests without global overrides or public test hooks. See [M05 — Conformance and D4 handoff](../../../../third-party/predicate-optimization/private-docs/Working/MilvusIntegration/PhaseD-ImplementationAndConformance/D3-SessionOrchestration/M05-ConformanceAndD4Handoff.md) for evidence and remaining D4, concurrent-operation and wired-route verification.
