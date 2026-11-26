# Translation contracts

`semantic_oracle_test.go` independently evaluates the admitted scalar test domain against explicit expected row results. It does not call adapter function implementations. `semantic_export_test.go` compares source, deliberately different target fixtures and actual output, including schema flags and cleared native template metadata.

`graph_export_test.go` exercises changed operators/operands, FALSE, balanced n-ary logic, sorted membership, graph/output ownership and atomic coverage failure. `export_expectations_test.go` converts historical source fixtures into expected executable form for structural regressions; it is not the independent semantic oracle.

Exact equality remains appropriate for source/schema immutability, independent output ownership and unchanged opaque payloads. Native source templates, specialized range shape and membership order are not the export contract.

The oracle fails on unsupported forms/operators and is not a backend emulator. Native execution, pruning, partition isolation and performance verification remain E05.

Files named `roundtrip_<variant>_test.go` cover paired import/export behavior, including import coverage cases. Direction-specific tests use `import_` or `export_` prefixes.
