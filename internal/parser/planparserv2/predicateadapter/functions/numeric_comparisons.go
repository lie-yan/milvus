package functions

import (
	"strconv"

	"github.com/lie-yan/predicate-optimization/foundation"
	shared "github.com/lie-yan/predicate-optimization/functions"
	"github.com/lie-yan/predicate-optimization/types"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/internal/assertions"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Declares six same-Type comparisons per numeric domain without admitting mixed-Type coercion.
func NumericComparisonDefinitions(boolean *types.BoolType, operands ...types.Type) ([]FunctionDefinition, error) {
	assertions.NotNilPointer(boolean)
	definitions := make([]FunctionDefinition, 0, 6*len(operands))
	for _, operand := range operands {
		assertions.NotNil(operand)
		var name string
		switch typ := operand.(type) {
		case *types.IntType:
			name = "i" + strconv.Itoa(int(typ.IntegerWidth()))
		case *types.FloatType:
			name = "f" + strconv.Itoa(int(typ.FloatWidth()))
		default:
			return nil, merr.WrapErrServiceInternalMsg("predicateadapter: numeric comparisons require numeric Types")
		}
		if operand.Nullability() != foundation.NonNull || boolean.Nullability() != foundation.NonNull {
			return nil, merr.WrapErrServiceInternalMsg("predicateadapter: numeric declarations require non-null Types")
		}
		for _, op := range []struct {
			suffix   string
			operator foundation.OperatorType
		}{
			{"eq", foundation.Equal}, {"ne", foundation.NotEqual}, {"lt", foundation.LessThan},
			{"le", foundation.LessThanOrEqual}, {"gt", foundation.GreaterThan}, {"ge", foundation.GreaterThanOrEqual},
		} {
			signature, err := shared.NewFunctionSignature([]types.Type{operand, operand}, boolean)
			if err != nil {
				return nil, err
			}
			metadata, err := shared.NewFunctionMetadata(signature, name+"."+op.suffix,
				foundation.Optional[shared.BuiltInFunctionName]{}, foundation.Some(op.operator),
				[]shared.NullPolicy{shared.PropagateNull, shared.PropagateNull})
			if err != nil {
				return nil, err
			}
			definitions = append(definitions, FunctionDefinition{Name: FunctionName{Namespace: "milvus.scalar.v1", Name: name + "." + op.suffix}, Metadata: metadata})
		}
	}
	return definitions, nil
}
