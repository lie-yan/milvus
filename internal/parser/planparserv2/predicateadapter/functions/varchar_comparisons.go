package functions

import (
	"github.com/lie-yan/predicate-optimization/foundation"
	shared "github.com/lie-yan/predicate-optimization/functions"
	"github.com/lie-yan/predicate-optimization/types"
)

// Declares equality and inequality for matching VARCHAR byte bounds, with null propagation.
func VarCharComparisonDeclarations(boolean *types.BoolType) ([]FunctionDeclaration, error) {
	pattern, err := types.NewVarCharTypePattern("x", foundation.NonNull)
	if err != nil {
		return nil, err
	}
	signature, err := shared.NewDeclarationSignature([]types.TypePattern{pattern, pattern}, types.NewConcreteTypePattern(boolean))
	if err != nil {
		return nil, err
	}
	declarations := make([]FunctionDeclaration, 0, 2)
	for _, spec := range []struct {
		name string
		op   foundation.OperatorType
	}{
		{"varchar.eq", foundation.Equal}, {"varchar.ne", foundation.NotEqual},
	} {
		declarations = append(declarations, FunctionDeclaration{
			Name: FunctionName{Namespace: "milvus.scalar.v1", Name: spec.name}, Signature: signature,
			DisplayName: spec.name, Operator: foundation.Some(spec.op),
			NullPolicies: []shared.NullPolicy{shared.PropagateNull, shared.PropagateNull},
		})
	}
	return declarations, nil
}
