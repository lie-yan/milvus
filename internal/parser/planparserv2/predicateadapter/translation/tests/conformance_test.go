package translation_test

import (
	"context"
	"testing"

	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/foundation"
	shared "github.com/lie-yan/predicate-optimization/functions"
	functioninterfaces "github.com/lie-yan/predicate-optimization/functions/interfaces"
	"github.com/lie-yan/predicate-optimization/types"
	valueinterfaces "github.com/lie-yan/predicate-optimization/values/interfaces"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/functions"
	adapterruntime "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/runtime"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

func TestCompositionOutputOwnership(t *testing.T) {
	ctx := context.Background()
	runtime, err := adapterruntime.SharedRuntime()
	require.NoError(t, err)
	translator, left, fields := compositionFixtureWithRuntime(t, &guardedRuntime{Runtime: runtime, provider: &noEvaluationContext{runtime.FunctionContext()}}, runtime.Types())
	right := proto.Clone(left).(*planpb.Expr)
	// Equal native graphs carry distinct source-occurrence metadata.
	rightParts := compositionParts(right)
	rightParts[1].GetUnaryRangeExpr().TemplateVariableName = "right_unary"
	rightParts[3].GetBinaryRangeExpr().LowerTemplateVariableName = "right_lower"
	rightParts[4].GetTermExpr().TemplateVariableName = "right_candidates"
	source := binarySource(planpb.BinaryExpr_LogicalAnd, left, right)
	sourceSnapshot := proto.Clone(source).(*planpb.Expr)
	schemaSnapshot := proto.Clone(&schemapb.CollectionSchema{Fields: fields})
	imported := importSource(t, translator, source)
	leftGraph, rightGraph := imported.Root().ChildAt(0), imported.Root().ChildAt(1)
	last := rightGraph
	for last.ChildCount() == 2 {
		last = last.ChildAt(1)
	}
	// Share every native subtree, retaining the right occurrence's opaque identity.
	rebuiltRight := replaceLastGraph(t, leftGraph, last)
	graph, err := expressions.NewLogicalExpression(foundation.And, []expressions.Expression{leftGraph, rebuiltRight})
	require.NoError(t, err)
	first, err := translator.Export(ctx, imported, graph)
	require.NoError(t, err)
	second, err := translator.Export(ctx, imported, graph)
	require.NoError(t, err)
	require.True(t, proto.Equal(expectedExport(source), first))

	parts := compositionParts(first.GetBinaryExpr().Left)
	parts[0].GetBinaryExpr().Right.GetUnaryExpr().Child.IsTemplate = true
	parts[1].GetUnaryRangeExpr().ColumnInfo.FieldId = 999
	parts[1].GetUnaryRangeExpr().Value.Val = &planpb.GenericValue_Int64Val{Int64Val: 999}
	parts[2].GetCompareExpr().RightColumnInfo.FieldId = 999
	parts[3].GetBinaryExpr().Left.GetUnaryRangeExpr().Value.Val = &planpb.GenericValue_FloatVal{FloatVal: 999}
	term := parts[4].GetTermExpr()
	term.Values[0].Val = &planpb.GenericValue_StringVal{StringVal: "changed"}
	term.Values[1] = stringSource("replacement")
	term.Values = append(term.Values, stringSource("appended"))
	parts[5].GetNullExpr().ColumnInfo.FieldId = 999

	require.True(t, proto.Equal(expectedExport(right), first.GetBinaryExpr().Right), "shared graph nodes must not share emitted messages")
	require.True(t, proto.Equal(sourceSnapshot, source), "borrowed source mutated")
	require.True(t, proto.Equal(schemaSnapshot, &schemapb.CollectionSchema{Fields: fields}), "pinned schema mutated")
	require.True(t, proto.Equal(expectedExport(sourceSnapshot), second), "independent export mutated")
	third, err := translator.Export(ctx, imported, graph)
	require.NoError(t, err)
	require.True(t, proto.Equal(expectedExport(sourceSnapshot), third), "candidate mutation reached import state")
}

// Counts context entry separately from invocation of a bound implementation.
type countingContext struct {
	functioninterfaces.FunctionContext
	evaluations int
}

func (c *countingContext) Evaluate(handle functioninterfaces.FunctionHandle, signature *shared.FunctionSignature, args []valueinterfaces.Value) (valueinterfaces.Value, error) {
	c.evaluations++
	return c.FunctionContext.Evaluate(handle, signature, args)
}

func TestCompleteTranslationDoesNotExecute(t *testing.T) {
	runtime, err := adapterruntime.SharedRuntime()
	require.NoError(t, err)
	p := runtime.Types()
	builder := functions.NewFunctionContextBuilder(runtime.Catalog())
	implementationCalls := 0
	for _, typ := range []types.Type{p.BoolNonNull, p.Int8NonNull, p.Int16NonNull, p.Int32NonNull, p.Int64NonNull, p.Float32NonNull, p.Float64NonNull} {
		ops := []planpb.OpType{planpb.OpType_Equal, planpb.OpType_NotEqual}
		if typ != p.BoolNonNull {
			ops = append(ops, planpb.OpType_LessThan, planpb.OpType_LessEqual, planpb.OpType_GreaterThan, planpb.OpType_GreaterEqual)
		}
		for _, op := range ops {
			handle, err := runtime.ComparisonFunction(op, typ, typ)
			require.NoError(t, err)
			require.NoError(t, builder.BindImplementation(handle, func([]valueinterfaces.Value) (valueinterfaces.Value, error) {
				implementationCalls++
				t.Fatal("translation invoked a function implementation")
				return nil, nil
			}))
		}
	}
	// Building a separate context leaves the published production bindings immutable.
	provider, err := builder.Build()
	require.NoError(t, err)
	counter := &countingContext{FunctionContext: provider}
	translator, source, _ := compositionFixtureWithRuntime(t, &guardedRuntime{Runtime: runtime, provider: counter}, p)
	imported := importSource(t, translator, source)
	ctx := context.Background()
	candidate, err := translator.Export(ctx, imported, imported.Root())
	require.NoError(t, err)
	require.True(t, proto.Equal(expectedExport(source), candidate))
	require.Zero(t, counter.evaluations)
	require.Zero(t, implementationCalls)
}
