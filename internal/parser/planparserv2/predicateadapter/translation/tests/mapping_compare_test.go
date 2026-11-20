package translation_test

import (
	"context"
	"testing"

	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/foundation"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	adapterruntime "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/runtime"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/schema"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/translation"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// Two independently typed fields and an evaluation-forbidden runtime.
func compareFixture(t *testing.T, left, right schemapb.DataType, nullable bool) (*translation.Translator, *planpb.Expr) {
	t.Helper()
	fields := []*schemapb.FieldSchema{
		{FieldID: 101, Name: "left", DataType: left},
		{FieldID: 102, Name: "right", DataType: right, Nullable: nullable},
	}
	for _, f := range fields {
		if f.DataType == schemapb.DataType_VarChar {
			f.TypeParams = []*commonpb.KeyValuePair{{Key: "max_length", Value: "100"}}
		}
	}
	helper, err := typeutil.CreateSchemaHelper(&schemapb.CollectionSchema{Fields: fields})
	require.NoError(t, err)
	view, err := schema.NewView(helper, fields)
	require.NoError(t, err)
	runtime, err := adapterruntime.SharedRuntime()
	require.NoError(t, err)
	translator, err := translation.New(&guardedRuntime{Runtime: runtime, provider: &noEvaluationContext{runtime.FunctionContext()}}, view, runtime.Types())
	require.NoError(t, err)
	return translator, &planpb.Expr{IsTemplate: true, Expr: &planpb.Expr_CompareExpr{CompareExpr: &planpb.CompareExpr{
		LeftColumnInfo:  &planpb.ColumnInfo{FieldId: 101, DataType: left},
		RightColumnInfo: &planpb.ColumnInfo{FieldId: 102, DataType: right, Nullable: nullable}, Op: planpb.OpType_Equal,
	}}}
}

func TestFieldComparisonMappings(t *testing.T) {
	ctx := context.Background()
	for _, typ := range []schemapb.DataType{schemapb.DataType_Bool, schemapb.DataType_Int8, schemapb.DataType_Int16, schemapb.DataType_Int32, schemapb.DataType_Int64, schemapb.DataType_Float, schemapb.DataType_Double} {
		t.Run(typ.String(), func(t *testing.T) {
			translator, source := compareFixture(t, typ, typ, false)
			ops := []planpb.OpType{planpb.OpType_Equal, planpb.OpType_NotEqual}
			if typ != schemapb.DataType_Bool {
				ops = append(ops, planpb.OpType_LessThan, planpb.OpType_LessEqual, planpb.OpType_GreaterThan, planpb.OpType_GreaterEqual)
			}
			for _, op := range ops {
				source.GetCompareExpr().Op = op
				for _, same := range []bool{false, true} {
					input := proto.Clone(source).(*planpb.Expr)
					if same {
						input.GetCompareExpr().RightColumnInfo = proto.Clone(input.GetCompareExpr().LeftColumnInfo).(*planpb.ColumnInfo)
					}
					_, imported := importSource(t, translator, input)
					call := imported.Root().(*expressions.CallExpression)
					left := call.ChildAt(0).(*expressions.ReferenceExpression)
					right := call.ChildAt(1).(*expressions.ReferenceExpression)
					require.Equal(t, same, left.ID().EqualsTo(right.ID()))
					// Same-field floating predicates must stay calls, including equality and inequality.
					rebuilt := rebuiltCall(t, call, left, right)
					output, err := translator.Export(ctx, imported, rebuilt)
					require.NoError(t, err)
					require.NoError(t, translator.ValidateCandidate(ctx, imported, output))
					require.True(t, proto.Equal(input, output))
				}
			}
		})
	}
}

func TestFieldComparisonAdmission(t *testing.T) {
	for _, tc := range []struct {
		left, right schemapb.DataType
		nullable    bool
	}{
		{schemapb.DataType_Int8, schemapb.DataType_Int64, false},
		{schemapb.DataType_Int64, schemapb.DataType_Int8, false},
		{schemapb.DataType_Float, schemapb.DataType_Double, false},
		{schemapb.DataType_Int64, schemapb.DataType_Double, false},
		{schemapb.DataType_VarChar, schemapb.DataType_VarChar, false},
		{schemapb.DataType_Int64, schemapb.DataType_Int64, true},
	} {
		translator, source := compareFixture(t, tc.left, tc.right, tc.nullable)
		admitted, err := translator.Admit(context.Background(), source)
		require.Nil(t, admitted)
		assertFailure(t, err, translation.UnsupportedSource, translation.StageAdmission)
	}
	for _, tc := range []struct {
		name   string
		change func(*planpb.CompareExpr)
		kind   translation.FailureKind
	}{
		{"missing right", func(n *planpb.CompareExpr) { n.RightColumnInfo = nil }, translation.UnsafeSource},
		{"right schema mismatch", func(n *planpb.CompareExpr) { n.RightColumnInfo.DataType = schemapb.DataType_Int32 }, translation.UnsafeSource},
		{"unsupported left cannot hide invalid right", func(n *planpb.CompareExpr) {
			n.LeftColumnInfo.NestedPath = []string{"x"}
			n.RightColumnInfo.FieldId = 999
		}, translation.UnsafeSource},
		{"invalid operator", func(n *planpb.CompareExpr) { n.Op = planpb.OpType_Invalid }, translation.UnsafeSource},
		{"future operator", func(n *planpb.CompareExpr) { n.Op = 999 }, translation.UnsupportedSource},
		{"unknown message", func(n *planpb.CompareExpr) { n.ProtoReflect().SetUnknown([]byte{0xa0, 0x06, 1}) }, translation.UnsupportedSource},
	} {
		t.Run(tc.name, func(t *testing.T) {
			translator, source := compareFixture(t, schemapb.DataType_Int64, schemapb.DataType_Int64, false)
			tc.change(source.GetCompareExpr())
			admitted, err := translator.Admit(context.Background(), binarySource(planpb.BinaryExpr_LogicalAnd, trueSource(), source))
			require.Nil(t, admitted)
			assertFailure(t, err, tc.kind, translation.StageAdmission)
		})
	}
	translator, source := compareFixture(t, schemapb.DataType_Bool, schemapb.DataType_Bool, false)
	source.GetCompareExpr().Op = planpb.OpType_LessThan
	_, err := translator.Admit(context.Background(), source)
	assertFailure(t, err, translation.UnsupportedSource, translation.StageAdmission)
}

func TestFieldComparisonCompositionAndDefects(t *testing.T) {
	ctx := context.Background()
	translator, source := compareFixture(t, schemapb.DataType_Int64, schemapb.DataType_Int64, false)
	_, imported := importSource(t, translator, source)
	call := imported.Root().(*expressions.CallExpression)
	_, foreign := importSource(t, translator, source)
	for _, args := range [][]expressions.Expression{
		{call.ChildAt(1), call.ChildAt(0)},
		{call.ChildAt(0), call.ChildAt(0)},
		{call.ChildAt(0), foreign.Root().ChildAt(1)},
	} {
		graph := rebuiltCall(t, call, args...)
		output, err := translator.Export(ctx, imported, graph)
		require.Nil(t, output)
		assertFailure(t, err, translation.TranslationDefect, translation.StageExport)
	}
	// A shared reference participates in both comparison forms under logical composition.
	unary := unarySource(source.GetCompareExpr().LeftColumnInfo, planpb.OpType_GreaterThan, integerSource(7))
	mixed := binarySource(planpb.BinaryExpr_LogicalAnd, source, unary)
	_, joined := importSource(t, translator, mixed)
	root := joined.Root()
	require.True(t, root.ChildAt(0).ChildAt(0).(*expressions.ReferenceExpression).ID().EqualsTo(root.ChildAt(1).ChildAt(0).(*expressions.ReferenceExpression).ID()))
	rebuilt := logicalGraph(t, foundation.And, root.ChildAt(0), root.ChildAt(1))
	output, err := translator.Export(ctx, joined, rebuilt)
	require.NoError(t, err)
	require.NoError(t, translator.ValidateCandidate(ctx, joined, output))
	require.True(t, proto.Equal(mixed, output))
	output.GetBinaryExpr().Left.GetCompareExpr().LeftColumnInfo.FieldId = 999
	require.Equal(t, int64(101), output.GetBinaryExpr().Right.GetUnaryRangeExpr().ColumnInfo.FieldId)
	again, err := translator.Export(ctx, joined, rebuilt)
	require.NoError(t, err)
	require.True(t, proto.Equal(mixed, again))
	assertFailure(t, translator.ValidateCandidate(ctx, joined, output), translation.TranslationDefect, translation.StageCandidateValidation)
	for _, change := range []func(*planpb.Expr){
		func(e *planpb.Expr) {
			n := e.GetCompareExpr()
			n.LeftColumnInfo, n.RightColumnInfo = n.RightColumnInfo, n.LeftColumnInfo
		},
		func(e *planpb.Expr) { e.GetCompareExpr().Op = planpb.OpType_NotEqual },
		func(e *planpb.Expr) { e.IsTemplate = false },
		func(e *planpb.Expr) { e.GetCompareExpr().ProtoReflect().SetUnknown([]byte{0xa0, 0x06, 1}) },
	} {
		candidate := proto.Clone(source).(*planpb.Expr)
		change(candidate)
		assertFailure(t, translator.ValidateCandidate(ctx, imported, candidate), translation.TranslationDefect, translation.StageCandidateValidation)
	}
}
