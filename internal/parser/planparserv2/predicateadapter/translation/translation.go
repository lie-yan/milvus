// Package translation owns paired expression translation and field/opaque bindings, independent of runtime publication and caller outcome policy.
package translation

import (
	functioninterfaces "github.com/lie-yan/predicate-optimization/functions/interfaces"
	"github.com/lie-yan/predicate-optimization/types"
	valueinterfaces "github.com/lie-yan/predicate-optimization/values/interfaces"

	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

// Stable immutable capabilities borrowed from one runtime for the entire translation attempt.
type Runtime interface {
	// Supplies the selected stateless factory without acquiring another runtime.
	ValueFactory() valueinterfaces.ValueFactory
	// Supplies the exact provider retained by imported calls.
	FunctionContext() functioninterfaces.FunctionContext
	// Resolves a Milvus comparison using this runtime's catalog and binding rules.
	ComparisonFunction(planpb.OpType, types.Type, types.Type) (functioninterfaces.FunctionHandle, error)
	// Recovers the Milvus comparison after validating handle ownership.
	ComparisonOperator(functioninterfaces.FunctionHandle) (planpb.OpType, error)
}
