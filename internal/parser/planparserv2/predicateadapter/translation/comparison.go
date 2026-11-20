package translation

import (
	"reflect"

	"github.com/lie-yan/predicate-optimization/expressions"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/schema"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

// Allocates one owning registry at the first admitted field-bearing import.
func (v *importer) ensureFields() error {
	if v.result.fields == nil {
		registry, err := schema.NewFieldRegistry(v.result.admitted.owner.view)
		if err != nil {
			return newFailure(TranslationDefect, StageImport, err)
		}
		v.result.fields = registry
	}
	return nil
}

// Checks actual call ownership, canonical operator, arity and result before form-specific operands.
func (v *graphChecker) checkComparisonCall(imported expressions.Expression, graph *expressions.CallExpression, expected planpb.OpType) error {
	original, ok := imported.(*expressions.CallExpression)
	if !ok || v.imported.fields == nil {
		return translationDefect(v.stage, "call has no comparison registry")
	}
	owner := v.imported.admitted.owner
	// Value comparability guards interface equality; production providers have pointer identity.
	provider := graph.Context()
	if !reflect.ValueOf(provider).Comparable() || provider != owner.context {
		return translationDefect(v.stage, "comparison changed function context")
	}
	if graph.ChildCount() != 2 || !owner.primitives.BoolNonNull.EqualsTo(graph.Type()) || !original.Handle().EqualsTo(graph.Handle()) {
		return translationDefect(v.stage, "comparison changed handle, result Type or arity")
	}
	op, err := owner.runtime.ComparisonOperator(graph.Handle())
	if err != nil {
		return newFailure(TranslationDefect, v.stage, err)
	}
	if op != expected {
		return translationDefect(v.stage, "comparison changed operator")
	}
	return nil
}
