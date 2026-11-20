package predicateadapter_test

import (
	"testing"

	"github.com/lie-yan/predicate-optimization/expressions"
	expressionutils "github.com/lie-yan/predicate-optimization/expressions/utils"
	shared "github.com/lie-yan/predicate-optimization/functions"
	valueinterfaces "github.com/lie-yan/predicate-optimization/values/interfaces"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/functions"
	adapterruntime "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/runtime"
	adapterschema "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/schema"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

func TestSharedFoundationComposition(t *testing.T) {
	runtime, err := adapterruntime.SharedRuntime()
	require.NoError(t, err)
	other, err := adapterruntime.SharedRuntime()
	require.NoError(t, err)
	require.Same(t, runtime, other)
	primitive := runtime.Types()
	handle, err := runtime.ComparisonFunction(planpb.OpType_GreaterThan, primitive.Int64NonNull, primitive.Int64NonNull)
	require.NoError(t, err)
	otherHandle, err := other.ComparisonFunction(planpb.OpType_GreaterThan, primitive.Int64NonNull, primitive.Int64NonNull)
	require.NoError(t, err)
	require.True(t, handle.EqualsTo(otherHandle))

	field := &schemapb.FieldSchema{FieldID: 101, Name: "i", DataType: schemapb.DataType_Int64}
	schema := &schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{field}}
	helper, err := typeutil.CreateSchemaHelper(schema)
	require.NoError(t, err)
	view, err := adapterschema.NewView(helper, schema.Fields)
	require.NoError(t, err)
	info := &planpb.ColumnInfo{FieldId: 101, DataType: schemapb.DataType_Int64}
	var registries []*adapterschema.FieldRegistry
	var references []*expressions.ReferenceExpression
	for range 2 {
		registry, err := adapterschema.NewFieldRegistry(view)
		require.NoError(t, err)
		binding, supported, err := registry.BindColumn(info, primitive.Int64NonNull)
		require.NoError(t, err)
		require.True(t, supported)
		reference, err := expressions.NewReferenceExpression(binding.ColumnID(), binding.Type(), field.Name)
		require.NoError(t, err)
		registries = append(registries, registry)
		references = append(references, reference)
	}
	require.False(t, references[0].ID().EqualsTo(references[1].ID()))

	// Instrument a separate context over the same catalog without replacing production bindings.
	builder := functions.NewFunctionContextBuilder(runtime.Catalog())
	implementationCalls := 0
	require.NoError(t, builder.BindImplementation(handle, func([]valueinterfaces.Value) (valueinterfaces.Value, error) {
		implementationCalls++
		t.Fatal("foundation composition invoked an implementation")
		return nil, nil
	}))
	instrumented, err := builder.Build()
	require.NoError(t, err)
	guards := []*noEvaluationContext{
		{FunctionContext: runtime.FunctionContext(), t: t},
		{FunctionContext: instrumented, t: t},
	}
	value, err := runtime.ValueFactory().NewInteger(primitive.Int64NonNull, 42)
	require.NoError(t, err)
	literal, err := expressions.NewLiteralExpression(value)
	require.NoError(t, err)
	for i, reference := range references {
		for _, guard := range guards {
			call, err := expressions.NewCallExpression(guard, handle, []expressions.Expression{reference, literal})
			require.NoError(t, err)
			require.Same(t, guard, call.Context())
			require.NoError(t, expressionutils.ValidateForOptimization(call))
			mapped, err := runtime.ComparisonOperator(call.Handle())
			require.NoError(t, err)
			require.Equal(t, planpb.OpType_GreaterThan, mapped)
		}
		resolved, err := registries[i].ResolveColumn(reference.ID(), reference.Type())
		require.NoError(t, err)
		require.Same(t, field, resolved.Field())
		_, err = registries[1-i].ResolveColumn(reference.ID(), reference.Type())
		require.ErrorIs(t, err, merr.ErrServiceInternal)
	}
	for _, guard := range guards {
		require.Zero(t, guard.evaluations)
	}
	require.Zero(t, implementationCalls)

	// Reuse S07's executable fixture; its catalog cannot be attached to runtime calls.
	harness := newComparisonHarness(t)
	_, err = expressions.NewCallExpression(harness.context, handle, []expressions.Expression{references[0], literal})
	require.ErrorIs(t, err, shared.ErrIncompatibleHandle)
	require.Zero(t, harness.calls)
}
