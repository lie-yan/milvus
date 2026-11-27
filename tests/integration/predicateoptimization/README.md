# Empty-pipeline execution verification

Run through Milvus Dev CLI:

```sh
milvus-dev-cli go-integration local . -b master --allow-dirty \
  -t tests/integration/predicateoptimization -k '^TestEmptyPipeline$'
```

The suite compares switch-off/on query/count, search, hybrid and scorer results. It checks actual scalar-index/scan execution and segment pruning through existing execution-feature counters. Clustering compaction supplies disjoint segment statistics for the pruning fixture.

## Observed server fixture

`TestMain` builds a temporary server using a Go compiler overlay. Only the call from `integration.Apply` to `orchestration.RoundTrip` is wrapped; normal source files and the normal binary remain unchanged. The wrapper calls the real round trip and publishes receipts containing the proxy PID, outcome, abort marker and error code. The build fails if that source seam changes. `probe.go.txt` is fixture input, not a production Go source file.

Each test arms exact deterministic protobuf hashes computed from the same collection schema and post-legacy parser output. A per-attempt exclusive claim makes observations one-shot; unrelated expressions cannot consume them. Control files and atomic JSON receipts live under the suite's temporary directory until its processes stop. Only proxy processes receive the control path. There is no production configuration key, global runtime override or public diagnostic API.

The observation action preserves the adapter input and output. The pause action blocks one selected attempt behind a bounded release barrier, allowing the real client RPC to be canceled before translation completes. The unsafe-source action substitutes an empty internal expression immediately before the real round trip: admission and orchestration must generate the real abort, and the actual query/search/scorer RPC boundary must preserve its system classification. Fault cases do not claim that an ordinary user can produce that malformed internal expression.

The uninstrumented binary's parity result is recorded separately in W08. The observed fixture adds evidence that enabled requests translate instead of merely retaining their inputs. No optimizer pass runs. Future rewrite tests should use separate semantic expectations without weakening these empty-pipeline checks.

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
