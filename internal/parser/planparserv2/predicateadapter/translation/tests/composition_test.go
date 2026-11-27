package translation_test

import (
	"context"
	"testing"

	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/foundation"
	shared "github.com/lie-yan/predicate-optimization/functions"
	functioninterfaces "github.com/lie-yan/predicate-optimization/functions/interfaces"
	"github.com/lie-yan/predicate-optimization/types"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	adapterruntime "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/runtime"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/schema"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/translation"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// Every implemented mapping in a mixed-domain tree, with preservation as the final region.
func compositionFixture(t *testing.T) (*translation.Translator, *planpb.Expr) {
	t.Helper()
	runtime, err := adapterruntime.SharedRuntime()
	require.NoError(t, err)
	translator, source, _ := compositionFixtureWithRuntime(t, &guardedRuntime{Runtime: runtime, provider: &noEvaluationContext{runtime.FunctionContext()}}, runtime.Types())
	return translator, source
}

func compositionFixtureWithRuntime(t *testing.T, runtime translation.Runtime, primitives adapterruntime.PrimitiveTypes) (*translation.Translator, *planpb.Expr, []*schemapb.FieldSchema) {
	t.Helper()
	fields := []*schemapb.FieldSchema{
		{FieldID: 101, Name: "integer", DataType: schemapb.DataType_Int64},
		{FieldID: 102, Name: "narrow_left", DataType: schemapb.DataType_Int16},
		{FieldID: 103, Name: "narrow_right", DataType: schemapb.DataType_Int16},
		{FieldID: 104, Name: "floating", DataType: schemapb.DataType_Float},
		{FieldID: 105, Name: "text", DataType: schemapb.DataType_VarChar, TypeParams: []*commonpb.KeyValuePair{{Key: "max_length", Value: "20"}}},
		{FieldID: 106, Name: "nullable", DataType: schemapb.DataType_Double, Nullable: true},
	}
	helper, err := typeutil.CreateSchemaHelper(&schemapb.CollectionSchema{Fields: fields})
	require.NoError(t, err)
	view, err := schema.NewView(helper, fields)
	require.NoError(t, err)
	translator, err := translation.New(runtime, view, primitives)
	require.NoError(t, err)
	columns := make([]*planpb.ColumnInfo, len(fields))
	for i, f := range fields {
		columns[i] = &planpb.ColumnInfo{FieldId: f.FieldID, DataType: f.DataType, Nullable: f.Nullable}
	}
	parts := []*planpb.Expr{
		binarySource(planpb.BinaryExpr_LogicalOr, trueSource(), notSource(trueSource())),
		unarySource(columns[0], planpb.OpType_GreaterEqual, integerSource(7)),
		{Expr: &planpb.Expr_CompareExpr{CompareExpr: &planpb.CompareExpr{LeftColumnInfo: columns[1], RightColumnInfo: columns[2], Op: planpb.OpType_NotEqual}}},
		rangeSource(columns[3], floatSource(0.5), floatSource(2)),
		termSource(columns[4], stringSource("z"), stringSource("a"), stringSource("z")),
		{Expr: &planpb.Expr_NullExpr{NullExpr: &planpb.NullExpr{ColumnInfo: columns[5], Op: planpb.NullExpr_IsNull}}},
	}
	root := parts[len(parts)-1]
	for i := len(parts) - 2; i >= 0; i-- {
		root = binarySource(planpb.BinaryExpr_LogicalAnd, parts[i], root)
	}
	return translator, root, fields
}

// Traverses the right-associated composition fixture without descending into each mapping.
func compositionParts(root *planpb.Expr) []*planpb.Expr {
	var parts []*planpb.Expr
	for root.GetBinaryExpr() != nil {
		parts = append(parts, root.GetBinaryExpr().Left)
		root = root.GetBinaryExpr().Right
	}
	return append(parts, root)
}

func replaceLastGraph(t *testing.T, root, last expressions.Expression) expressions.Expression {
	t.Helper()
	if node, ok := root.(*expressions.LogicalExpression); ok && node.Operator() == foundation.And {
		rebuilt, err := expressions.NewLogicalExpression(node.Operator(), []expressions.Expression{node.ChildAt(0), replaceLastGraph(t, node.ChildAt(1), last)})
		require.NoError(t, err)
		return rebuilt
	}
	return last
}

func TestAllMappingsCompose(t *testing.T) {
	translator, source := compositionFixture(t)
	imported := importSource(t, translator, source)
	graph := imported.Root()
	// Native forms must remain exposed rather than succeeding through opaque copies.
	for _, expected := range []any{&expressions.LogicalExpression{}, &expressions.CallExpression{}, &expressions.CallExpression{}, &expressions.LogicalExpression{}, &expressions.SpecialFormExpression{}} {
		require.IsType(t, expected, graph.ChildAt(0))
		graph = graph.ChildAt(1)
	}
	require.IsType(t, &expressions.OpaqueExpression{}, graph)
	output, err := translator.Export(context.Background(), imported, imported.Root())
	require.NoError(t, err)
	require.True(t, proto.Equal(expectedExport(source), output))
	parts := compositionParts(output)
	require.Equal(t, planpb.OpType_NotEqual, parts[2].GetCompareExpr().Op)
	require.Equal(t, float64(0.5), parts[3].GetBinaryExpr().Left.GetUnaryRangeExpr().Value.GetFloatVal())
	require.Equal(t, []string{"a", "z", "z"}, []string{parts[4].GetTermExpr().Values[0].GetStringVal(), parts[4].GetTermExpr().Values[1].GetStringVal(), parts[4].GetTermExpr().Values[2].GetStringVal()})
	require.Equal(t, planpb.NullExpr_IsNull, parts[5].GetNullExpr().Op)
}

func TestCompositionImportCoverageAndLateGraphFailure(t *testing.T) {
	ctx := context.Background()
	translator, source := compositionFixture(t)
	for _, tc := range []struct {
		name        string
		replacement *planpb.Expr
		kind        translation.FailureKind
	}{
		{"no route", &planpb.Expr{Expr: &planpb.Expr_CallExpr{CallExpr: &planpb.CallExpr{FunctionName: "unknown"}}}, translation.UnsupportedExpression},
		{"invalid source", &planpb.Expr{Expr: &planpb.Expr_TermExpr{TermExpr: &planpb.TermExpr{}}}, translation.UnsafeSource},
	} {
		t.Run(tc.name, func(t *testing.T) {
			changed := proto.Clone(source).(*planpb.Expr)
			last := changed
			for last.GetBinaryExpr().Right.GetBinaryExpr() != nil {
				last = last.GetBinaryExpr().Right
			}
			last.GetBinaryExpr().Right = tc.replacement
			imported, err := translator.Import(ctx, changed)
			require.Nil(t, imported)
			assertFailure(t, err, tc.kind, translation.StageImport)
		})
	}
	// Stop at unsupported coverage, even when a later sibling is malformed.
	changed := proto.Clone(source).(*planpb.Expr)
	last := changed
	for last.GetBinaryExpr().Right.GetBinaryExpr() != nil {
		last = last.GetBinaryExpr().Right
	}
	last.GetBinaryExpr().Right = binarySource(planpb.BinaryExpr_LogicalOr, &planpb.Expr{Expr: &planpb.Expr_CallExpr{CallExpr: &planpb.CallExpr{}}}, &planpb.Expr{})
	partial, err := translator.Import(ctx, changed)
	require.Nil(t, partial)
	assertFailure(t, err, translation.UnsupportedExpression, translation.StageImport)
	imported := importSource(t, translator, source)
	foreign := importSource(t, translator, source)
	foreignLeaf := foreign.Root()
	for foreignLeaf.ChildCount() == 2 {
		foreignLeaf = foreignLeaf.ChildAt(1)
	}
	originalLeaf := imported.Root()
	for originalLeaf.ChildCount() == 2 {
		originalLeaf = originalLeaf.ChildAt(1)
	}
	runtime, err := adapterruntime.SharedRuntime()
	require.NoError(t, err)
	wrongType, err := expressions.NewOpaqueExpression(originalLeaf.(*expressions.OpaqueExpression).ID(), runtime.Types().BoolMayNull, "changed")
	require.NoError(t, err)
	for _, leaf := range []expressions.Expression{foreignLeaf, wrongType} {
		bad := replaceLastGraph(t, imported.Root(), leaf)
		output, err := translator.Export(ctx, imported, bad)
		require.Nil(t, output)
		assertFailure(t, err, translation.TranslationDefect, translation.StageExport)
	}
}

// Visits participating protobuf messages, including embedded operands and repeated candidates.
func visitMessages(message protoreflect.Message, visit func(protoreflect.Message)) {
	visit(message)
	message.Range(func(field protoreflect.FieldDescriptor, value protoreflect.Value) bool {
		if field.Message() != nil {
			if field.IsList() {
				list := value.List()
				for i := 0; i < list.Len(); i++ {
					visitMessages(list.Get(i).Message(), visit)
				}
			} else {
				visitMessages(value.Message(), visit)
			}
		}
		return true
	})
}

func TestCompositionUnknownFields(t *testing.T) {
	translator, source := compositionFixture(t)
	// One occurrence of every participating message type is enough to exercise its coverage gate.
	seen := map[protoreflect.FullName]bool{}
	var names []protoreflect.FullName
	visitMessages(source.ProtoReflect(), func(m protoreflect.Message) {
		name := m.Descriptor().FullName()
		// Unary comparisons trust the locally produced message layout.
		if _, ok := m.Interface().(*planpb.UnaryRangeExpr); ok {
			return
		}
		if !seen[name] {
			seen[name] = true
			names = append(names, name)
		}
	})
	for _, name := range names {
		t.Run(string(name), func(t *testing.T) {
			mutated := proto.Clone(source).(*planpb.Expr)
			changed := false
			visitMessages(mutated.ProtoReflect(), func(m protoreflect.Message) {
				if !changed && m.Descriptor().FullName() == name {
					m.SetUnknown([]byte{0xa0, 0x06, 1})
					changed = true
				}
			})
			imported, err := translator.Import(context.Background(), mutated)
			require.Nil(t, imported)
			assertFailure(t, err, translation.UnsupportedExpression, translation.StageImport)
		})
	}
}

// Injects a missing wider overload while keeping the original eight signatures intact.
type missingNumericRuntime struct{ translation.Runtime }

func (r missingNumericRuntime) ComparisonFunction(op planpb.OpType, left, right types.Type) (functioninterfaces.FunctionHandle, error) {
	if _, ok := left.(*types.FloatType); ok {
		return nil, shared.ErrUnsupportedLookup
	}
	return r.Runtime.ComparisonFunction(op, left, right)
}

func TestWiderRuntimeSetupContract(t *testing.T) {
	runtime, err := adapterruntime.SharedRuntime()
	require.NoError(t, err)
	helper, err := typeutil.CreateSchemaHelper(&schemapb.CollectionSchema{})
	require.NoError(t, err)
	view, err := schema.NewView(helper, nil)
	require.NoError(t, err)
	translator, err := translation.New(missingNumericRuntime{runtime}, view, runtime.Types())
	require.Nil(t, translator)
	require.ErrorIs(t, err, shared.ErrUnsupportedLookup)
	// A mislabeled slot would otherwise bind a reference with the wrong declared width.
	primitives := runtime.Types()
	primitives.Int16NonNull = primitives.Int8NonNull
	translator, err = translation.New(runtime, view, primitives)
	require.Nil(t, translator)
	require.Error(t, err)
}
