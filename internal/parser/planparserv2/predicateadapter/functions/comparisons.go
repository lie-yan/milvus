package functions

import (
	"github.com/lie-yan/predicate-optimization/foundation"
	shared "github.com/lie-yan/predicate-optimization/functions"
	"github.com/lie-yan/predicate-optimization/types"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/internal/assertions"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Builds the initial eight comparison declarations using shared non-null Int64/Bool Types.
// These registrations define the initial runtime surface, not catalog or translator type limits.
func ComparisonDefinitions(integer *types.IntType, boolean *types.BoolType) ([]FunctionDefinition, error) {
	assertions.NotNilPointer(integer)
	assertions.NotNilPointer(boolean)
	if integer.IntegerWidth() != types.Int64 || integer.Nullability() != foundation.NonNull || boolean.Nullability() != foundation.NonNull {
		return nil, merr.WrapErrServiceInternalMsg("predicateadapter: initial comparisons require non-null Int64 and Bool Types")
	}
	definitions := make([]FunctionDefinition, 0, 8)
	for _, spec := range []struct {
		name    string
		op      foundation.OperatorType
		operand types.Type
	}{
		{"i64.eq", foundation.Equal, integer},
		{"i64.ne", foundation.NotEqual, integer},
		{"i64.lt", foundation.LessThan, integer},
		{"i64.le", foundation.LessThanOrEqual, integer},
		{"i64.gt", foundation.GreaterThan, integer},
		{"i64.ge", foundation.GreaterThanOrEqual, integer},
		{"bool.eq", foundation.Equal, boolean},
		{"bool.ne", foundation.NotEqual, boolean},
	} {
		signature, err := shared.NewFunctionSignature([]types.Type{spec.operand, spec.operand}, boolean)
		if err != nil {
			return nil, merr.WrapErrServiceInternalErr(err, "predicateadapter: invalid comparison signature")
		}
		metadata, err := shared.NewFunctionMetadata(signature, spec.name,
			foundation.Optional[shared.BuiltInFunctionName]{}, foundation.Some(spec.op),
			[]shared.NullPolicy{shared.PropagateNull, shared.PropagateNull})
		if err != nil {
			return nil, merr.WrapErrServiceInternalErr(err, "predicateadapter: invalid comparison metadata")
		}
		definitions = append(definitions, FunctionDefinition{
			Name: FunctionName{Namespace: "milvus.scalar.v1", Name: spec.name}, Metadata: metadata,
		})
	}
	return definitions, nil
}
