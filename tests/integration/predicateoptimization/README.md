# Empty-pipeline execution verification

Run through Milvus Dev CLI:

```sh
milvus-dev-cli go-integration local . -b master --allow-dirty \
  -t tests/integration/predicateoptimization -k '^TestEmptyPipeline$'
```

The suite compares switch-off/on query/count, search, hybrid and scorer results. It checks actual scalar-index/scan execution and segment pruning through existing execution-feature counters. Clustering compaction supplies disjoint segment statistics for the pruning fixture.

## Observed server fixture

`TestMain` builds a temporary server using a Go compiler overlay. The call from `integration.Apply` to `orchestration.Optimize` is wrapped; normal source files and the normal binary remain unchanged. The wrapper calls the real round trip and publishes receipts containing the proxy PID, outcome, abort marker and error code. The build fails if that source seam changes. `probe.go.txt` is fixture input, not a production Go source file.

Each test arms exact deterministic protobuf hashes computed from the same collection schema and post-legacy parser output. A per-attempt exclusive claim makes observations one-shot; unrelated expressions cannot consume them. Control files and atomic JSON receipts live under the suite's temporary directory until its processes stop. Only proxy processes receive the control path. There is no production configuration key, global runtime override or public diagnostic API.

The observation action preserves the adapter input and output. The pause action blocks one selected attempt behind a bounded release barrier, allowing the real client RPC to be canceled before translation completes. The unsafe-source action substitutes an empty internal expression immediately before the real round trip: admission and orchestration must generate the real abort, and the actual query/search/scorer RPC boundary must preserve its system classification. Fault cases do not claim that an ordinary user can produce that malformed internal expression.

The uninstrumented binary's parity result is recorded separately in W08. The observed fixture adds evidence that enabled requests translate instead of merely retaining their inputs. Existing empty-pipeline cases select zero passes through the orchestration overlay. Separate optimized cases use the same independent semantic expectations without weakening the empty-pipeline checks.

## Complete touchstone gate

`TestEmptyPipeline/TestTouchstoneExecutionParity` uses the shared fixture under
`internal/parser/planparserv2/predicateadapter/tests/touchstone`. It checks the
primary and nullable-article schemas with the switch off/on, before and after
INVERTED scalar indexes. Queries for P and NOT P establish TRUE/FALSE/UNKNOWN
against the recorded independent expectations. Enabled requests require an
OutcomeTranslated receipt and unchanged source; an array probe confirms the
native scan/index path. This test uses zero optimizer passes.

The corresponding local parser test, `TestPredicateOptimizationTouchstoneWiring`,
checks both legacy rewrite settings, requires every touchstone operation to be
exposed (no opaque leaves), and forbids evaluation. Native results must be recorded
before declaring the touchstone empty-pipeline gate passed.

## Optimized touchstone gate

`TestTouchstoneOptimizedExecutionParity` repeats the touchstone matrix after each cumulative prefix of the five passes. The orchestration overlay selects the prefix from the exact source control and records `rewritten.json`; count 5 calls the production rewrite function. P/NOT P and UNKNOWN expectations remain independent of graph shape. `TestOptimizedExecutionParity` also runs the existing query/count/search/hybrid/scorer regression with all five passes. Unmatched/ordinary observation actions retain the explicit zero-pass baseline in this test server only; normal production uses all five passes when the adapter switch is enabled.

```sh
milvus-dev-cli go-integration local . -b master --allow-dirty \
  -t tests/integration/predicateoptimization \
  -k '^TestEmptyPipeline$/^(TestTouchstoneOptimizedExecutionParity|TestTouchstoneExecutionParity|TestOptimizedExecutionParity|TestExecutionParity)$'
```

## Optimized delete parity

`TestOptimizedDeleteParity` compares native Delete effects on fresh identical
24-row collections with the switch off/on. It covers primary-key deletion,
nullable filtering, the full simplifying touchstone and unsupported arithmetic.
Independent expectations check mutation counts and Strong-consistency survivor
IDs. Blank/true/Bloom rejection must leave rows intact; the enabled compound
tautology must translate before always-true rejection. Receipts establish actual
translation/retention and touchstone simplification. Scope is flushed/loaded data
without explicit scalar indexes, not concurrent or growing-segment parity.

```sh
milvus-dev-cli go-integration local . -b master --allow-dirty \
  -t tests/integration/predicateoptimization \
  -k '^TestEmptyPipeline$/^TestOptimizedDeleteParity$'
```
