package translation

import (
	"cmp"
	"slices"

	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/foundation"

	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

// Emits homogeneous scalar membership with a fresh sorted candidate list for native pruning.
func (v *exporter) exportTerm(graph *expressions.SpecialFormExpression) (*planpb.Expr, error) {
	if graph.Form() != foundation.SpecialFormIn || graph.Type().Nullability() != foundation.NonNull || graph.ChildCount() < 2 {
		return nil, unsupportedExport("membership form or domain")
	}
	target := graph.ChildAt(0)
	if _, ok := target.(*expressions.ReferenceExpression); !ok {
		return nil, unsupportedExport("membership requires a field target")
	}
	column, err := exportField(v.imported.fields, target)
	if err != nil {
		return nil, err
	}
	values := make([]*planpb.GenericValue, graph.ChildCount()-1)
	for i := range values {
		if err := v.ctx.Err(); err != nil {
			return nil, err
		}
		candidate := graph.ChildAt(i + 1)
		if !target.Type().EqualsTo(candidate.Type()) {
			return nil, unsupportedExport("heterogeneous membership")
		}
		values[i], err = exportScalar(candidate)
		if err != nil {
			return nil, err
		}
	}
	slices.SortStableFunc(values, compareScalars)
	return &planpb.Expr{Expr: &planpb.Expr_TermExpr{TermExpr: &planpb.TermExpr{ColumnInfo: column, Values: values}}}, nil
}

// Orders admitted homogeneous nonnull scalar values; stable sorting preserves signed-zero payload order.
func compareScalars(a, b *planpb.GenericValue) int {
	switch value := a.Val.(type) {
	case *planpb.GenericValue_BoolVal:
		if value.BoolVal == b.GetBoolVal() {
			return 0
		}
		if !value.BoolVal {
			return -1
		}
		return 1
	case *planpb.GenericValue_Int64Val:
		return cmp.Compare(value.Int64Val, b.GetInt64Val())
	case *planpb.GenericValue_FloatVal:
		return cmp.Compare(value.FloatVal, b.GetFloatVal())
	case *planpb.GenericValue_StringVal:
		return cmp.Compare(value.StringVal, b.GetStringVal())
	default:
		panic("unreachable scalar domain")
	}
}
