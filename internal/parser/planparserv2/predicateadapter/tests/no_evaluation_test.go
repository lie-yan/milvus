package predicateadapter_test

import (
	"testing"

	"github.com/lie-yan/predicate-optimization/expressions"
	expressionutils "github.com/lie-yan/predicate-optimization/expressions/utils"
	"github.com/lie-yan/predicate-optimization/foundation"
	shared "github.com/lie-yan/predicate-optimization/functions"
	functioninterfaces "github.com/lie-yan/predicate-optimization/functions/interfaces"
	functionutils "github.com/lie-yan/predicate-optimization/functions/utils"
	"github.com/lie-yan/predicate-optimization/types"
	valueinterfaces "github.com/lie-yan/predicate-optimization/values/interfaces"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/functions"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/values"
)

// Guards the interface entry independently of implementation dispatch, including NULL short-circuits.
type noEvaluationContext struct {
	functioninterfaces.FunctionContext
	t           *testing.T
	evaluations int
}

func (c *noEvaluationContext) Evaluate(functioninterfaces.FunctionHandle, *shared.FunctionSignature, []valueinterfaces.Value) (valueinterfaces.Value, error) {
	c.evaluations++
	c.t.Fatal("metadata composition invoked Evaluate")
	return nil, nil
}

func TestFunctionCompositionDoesNotEvaluate(t *testing.T) {
	int64NonNullType, err := types.NewIntType(types.Int64, foundation.NonNull)
	require.NoError(t, err)
	int64MayNullType, err := types.NewIntType(types.Int64, foundation.MayNull)
	require.NoError(t, err)
	boolNonNullType, err := types.NewBoolType(foundation.NonNull)
	require.NoError(t, err)
	definitions, err := functions.ComparisonDefinitions(int64NonNullType, boolNonNullType)
	require.NoError(t, err)
	catalogBuilder := functions.NewFunctionCatalogBuilder()
	for _, definition := range definitions {
		require.NoError(t, catalogBuilder.Register(definition))
	}
	catalog, err := catalogBuilder.Build()
	require.NoError(t, err)
	builder := functions.NewFunctionContextBuilder(catalog)
	implementationCalls := 0
	for _, definition := range definitions {
		handle, err := catalog.LookupFunction(definition.Name, definition.Metadata.Signature().ArgumentTypes())
		require.NoError(t, err)
		require.NoError(t, builder.BindImplementation(handle, func([]valueinterfaces.Value) (valueinterfaces.Value, error) {
			implementationCalls++
			t.Fatal("metadata composition invoked a function implementation")
			return nil, nil
		}))
	}
	context, err := builder.Build()
	require.NoError(t, err)
	guard := &noEvaluationContext{FunctionContext: context, t: t}
	integer, err := values.Factory.NewInteger(int64NonNullType, 42)
	require.NoError(t, err)
	boolean, err := values.Factory.NewBool(boolNonNullType, true)
	require.NoError(t, err)
	null, err := values.Factory.NewNull(int64MayNullType)
	require.NoError(t, err)
	var children []expressions.Expression
	for _, value := range []valueinterfaces.Value{integer, boolean, null} {
		typ := value.Type()
		handle, err := guard.ComparisonFunction(foundation.Equal, typ, typ)
		require.NoError(t, err)
		metadata, err := functionutils.InspectFunction(guard, handle)
		require.NoError(t, err)
		signature, err := shared.BindFunctionSignature(metadata, []types.Type{typ, typ})
		require.NoError(t, err)
		literal, err := expressions.NewLiteralExpression(value)
		require.NoError(t, err)
		call, err := expressions.NewCallExpression(guard, handle, []expressions.Expression{literal, literal})
		require.NoError(t, err)
		require.True(t, signature.ResultType().EqualsTo(call.Type()))
		require.Same(t, guard, call.Context())
		children = append(children, call)
	}
	root, err := expressions.NewLogicalExpression(foundation.And, children)
	require.NoError(t, err)
	require.NoError(t, expressionutils.Validate(root))
	require.NoError(t, expressionutils.ValidateForOptimization(root))
	require.Zero(t, guard.evaluations)
	require.Zero(t, implementationCalls)
}
