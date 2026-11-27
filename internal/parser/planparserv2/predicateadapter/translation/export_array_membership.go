package translation

import (
	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/foundation"
	shared "github.com/lie-yan/predicate-optimization/functions"
	"github.com/lie-yan/predicate-optimization/types"
	valueinterfaces "github.com/lie-yan/predicate-optimization/values/interfaces"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

// Reconstructs containment from the current graph, preserving candidate order and duplicates.
func (v *exporter) exportArrayMembership(call *expressions.CallExpression, name shared.BuiltInFunctionName) (*planpb.Expr, error) {
	target := call.ChildAt(0)
	array, ok := target.Type().(*types.ArrayType)
	if !ok || array.Element().Nullability() != foundation.NonNull {
		return nil, unsupportedExport("array containment target domain")
	}
	if _, ok := target.(*expressions.ReferenceExpression); !ok {
		return nil, unsupportedExport("array containment requires a field target")
	}
	column, err := exportField(v.imported.fields, target)
	if err != nil {
		return nil, err
	}
	if column.DataType != schemapb.DataType_Array {
		return nil, unsupportedExport("array containment requires a physical array")
	}
	literal, ok := call.ChildAt(1).(*expressions.LiteralExpression)
	if !ok || literal.Value().IsNull() || literal.Type().Nullability() != foundation.NonNull {
		return nil, unsupportedExport("array containment requires nonnull literal candidates")
	}
	n := &planpb.JSONContainsExpr{ColumnInfo: column, ElementsSameType: true}
	if name == shared.BuiltInArrayContains {
		if !literal.Type().EqualsTo(array.Element()) {
			return nil, unsupportedExport("array containment candidate domain")
		}
		value, err := exportScalarValue(literal.Value())
		if err != nil {
			return nil, err
		}
		n.Op, n.Elements = planpb.JSONContainsExpr_Contains, []*planpb.GenericValue{value}
	} else {
		if !literal.Type().EqualsToIgnoringNullability(array) {
			return nil, unsupportedExport("array containment candidate array domain")
		}
		values := literal.Value().(valueinterfaces.ArrayValue)
		n.Op = planpb.JSONContainsExpr_ContainsAny
		n.Elements = make([]*planpb.GenericValue, values.ElementCount())
		for i := range n.Elements {
			if err := v.ctx.Err(); err != nil {
				return nil, err
			}
			n.Elements[i], err = exportScalarValue(values.ElementAt(i))
			if err != nil {
				return nil, err
			}
		}
	}
	return &planpb.Expr{Expr: &planpb.Expr_JsonContainsExpr{JsonContainsExpr: n}}, nil
}
