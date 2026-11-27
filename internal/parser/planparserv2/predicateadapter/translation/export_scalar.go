package translation

import (
	"math"

	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/foundation"
	valueinterfaces "github.com/lie-yan/predicate-optimization/values/interfaces"

	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

// Allocates a scalar protobuf from the current immutable Value without evaluation or coercion.
func exportScalar(graph expressions.Expression) (*planpb.GenericValue, error) {
	literal, ok := graph.(*expressions.LiteralExpression)
	if !ok {
		return nil, unsupportedExport("expected nonnull scalar literal")
	}
	return exportScalarValue(literal.Value())
}

// Copies one exact nonnull scalar payload, including array-literal elements.
func exportScalarValue(value valueinterfaces.Value) (*planpb.GenericValue, error) {
	if value.Type().Nullability() != foundation.NonNull || value.IsNull() {
		return nil, unsupportedExport("expected nonnull scalar value")
	}
	result := &planpb.GenericValue{}
	switch value := value.(type) {
	case valueinterfaces.BoolValue:
		result.Val = &planpb.GenericValue_BoolVal{BoolVal: value.Bool()}
	case valueinterfaces.IntegerValue:
		result.Val = &planpb.GenericValue_Int64Val{Int64Val: value.Integer()}
	case valueinterfaces.FloatValue:
		if math.IsNaN(value.Float()) {
			return nil, unsupportedExport("NaN scalar literal")
		}
		result.Val = &planpb.GenericValue_FloatVal{FloatVal: value.Float()}
	case valueinterfaces.VarCharValue:
		result.Val = &planpb.GenericValue_StringVal{StringVal: value.VarChar()}
	default:
		return nil, unsupportedExport("scalar literal domain")
	}
	return result, nil
}
