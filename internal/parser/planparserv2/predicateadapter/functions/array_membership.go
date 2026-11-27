package functions

import (
	"github.com/lie-yan/predicate-optimization/foundation"
	shared "github.com/lie-yan/predicate-optimization/functions"
	"github.com/lie-yan/predicate-optimization/types"

	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Declares containment over concrete scalar element domains, with nullable arrays and nonnull candidates.
func ArrayMembershipDefinitions(boolean *types.BoolType, elements ...types.Type) ([]FunctionDefinition, error) {
	definitions := make([]FunctionDefinition, 0, 2*len(elements))
	for _, element := range elements {
		if element.Nullability() != foundation.NonNull {
			return nil, merr.WrapErrServiceInternalMsg("predicateadapter: array membership requires nonnull elements")
		}
		switch element.(type) {
		case *types.BoolType, *types.IntType, *types.FloatType, *types.VarCharType:
		default:
			return nil, merr.WrapErrServiceInternalMsg("predicateadapter: array membership requires scalar elements")
		}
		array, err := types.NewArrayType(element, foundation.NonNull)
		if err != nil {
			return nil, err
		}
		for _, spec := range []struct {
			name      shared.BuiltInFunctionName
			candidate types.Type
		}{
			{shared.BuiltInArrayContains, element},
			{shared.BuiltInArrayContainsAny, array},
		} {
			signature, err := shared.NewFunctionSignature([]types.Type{array, spec.candidate}, boolean)
			if err != nil {
				return nil, err
			}
			name := element.String() + "." + string(spec.name)
			metadata, err := shared.NewFunctionMetadata(signature, name, foundation.Some(spec.name), foundation.Optional[foundation.OperatorType]{}, []shared.NullPolicy{shared.PropagateNull, shared.NullPolicyUnspecified})
			if err != nil {
				return nil, err
			}
			definitions = append(definitions, FunctionDefinition{Name: FunctionName{Namespace: "milvus.array.v1", Name: name}, Metadata: metadata})
		}
	}
	return definitions, nil
}
