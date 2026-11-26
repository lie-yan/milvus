package translation_test

import (
	"context"
	"testing"

	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/foundation"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/translation"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

func TestLogicalMixedRoundTrip(t *testing.T) {
	ctx := context.Background()
	translator, opaque, _ := opaqueFixture(t, schemapb.DataType_VarChar, true)
	opaque.IsTemplate = true
	sharedTrue := trueSource()
	falseExpr := notSource(sharedTrue)
	falseExpr.IsTemplate = true
	distinctTrue := trueSource()
	distinctTrue.IsTemplate = true
	source := binarySource(planpb.BinaryExpr_LogicalAnd,
		binarySource(planpb.BinaryExpr_LogicalOr, opaque, falseExpr),
		binarySource(planpb.BinaryExpr_LogicalAnd, opaque, notSource(notSource(distinctTrue))))
	source.IsTemplate = true
	imported := importSource(t, translator, source)
	root := imported.Root().(*expressions.LogicalExpression)
	require.Equal(t, foundation.And, root.Operator())
	left := root.ChildAt(0).(*expressions.LogicalExpression)
	right := root.ChildAt(1).(*expressions.LogicalExpression)
	require.Equal(t, foundation.Or, left.Operator())
	require.Equal(t, foundation.And, right.Operator())
	require.Equal(t, foundation.Not, left.ChildAt(1).(*expressions.LogicalExpression).Operator())
	require.Equal(t, foundation.Not, right.ChildAt(1).(*expressions.LogicalExpression).ChildAt(0).(*expressions.LogicalExpression).Operator())
	require.False(t, left.ChildAt(0).(*expressions.OpaqueExpression).ID().EqualsTo(right.ChildAt(0).(*expressions.OpaqueExpression).ID()))
	// Registered opaque leaves may be shared across source positions.
	sharedOpaque := logicalGraph(t, foundation.And,
		logicalGraph(t, foundation.Or, right.ChildAt(0), left.ChildAt(1)), right)
	sharedOutput, err := translator.Export(ctx, imported, sharedOpaque)
	require.NoError(t, err)
	require.True(t, proto.Equal(expectedExport(source), sharedOutput))
	// Rebuild the root and share an equivalent native TRUE leaf at positions with different metadata.
	sharedLeaf := left.ChildAt(1).ChildAt(0)
	rebuilt := logicalGraph(t, foundation.And, left,
		logicalGraph(t, foundation.And, right.ChildAt(0), logicalGraph(t, foundation.Not, logicalGraph(t, foundation.Not, sharedLeaf))))
	candidate, err := translator.Export(ctx, imported, rebuilt)
	require.NoError(t, err)
	require.True(t, proto.Equal(expectedExport(source), candidate))
	second, err := translator.Export(ctx, imported, rebuilt)
	require.NoError(t, err)
	b := candidate.GetBinaryExpr()
	opaqueLeft := b.Left.GetBinaryExpr().Left
	opaqueRight := b.Right.GetBinaryExpr().Left
	require.NotSame(t, opaqueLeft.GetNullExpr().ColumnInfo, opaqueRight.GetNullExpr().ColumnInfo)
	firstTrue := b.Left.GetBinaryExpr().Right.GetUnaryExpr().Child
	secondTrue := b.Right.GetBinaryExpr().Right.GetUnaryExpr().Child.GetUnaryExpr().Child
	require.NotSame(t, firstTrue, secondTrue)
	require.NotSame(t, firstTrue.GetAlwaysTrueExpr(), secondTrue.GetAlwaysTrueExpr())
	firstTrue.IsTemplate = true
	opaqueLeft.GetNullExpr().ColumnInfo.NestedPath = []string{"changed"}
	require.True(t, proto.Equal(expectedExport(source), second))
	require.Empty(t, opaqueRight.GetNullExpr().ColumnInfo.NestedPath)
	// Root constants use the same native routes, preserving FALSE as NOT TRUE.
	for _, source := range []*planpb.Expr{trueSource(), notSource(trueSource()), binarySource(planpb.BinaryExpr_LogicalAnd, sharedTrue, sharedTrue)} {
		imported := importSource(t, translator, source)
		candidate, err := translator.Export(ctx, imported, imported.Root())
		require.NoError(t, err)
		require.True(t, proto.Equal(expectedExport(source), candidate))
	}
}

func TestLogicalImportFailures(t *testing.T) {
	translator, opaque, _ := opaqueFixture(t, schemapb.DataType_Bool, true)
	unknown := &planpb.Expr{Expr: &planpb.Expr_CallExpr{CallExpr: &planpb.CallExpr{FunctionName: "unknown"}}}
	malformed := &planpb.Expr{Expr: &planpb.Expr_NullExpr{NullExpr: &planpb.NullExpr{}}}
	future := binarySource(99, trueSource(), opaque)
	badOp := notSource(trueSource())
	badOp.GetUnaryExpr().Op = planpb.UnaryExpr_Invalid
	badBinary := binarySource(planpb.BinaryExpr_Invalid, trueSource(), opaque)
	unknownTrue := trueSource()
	unknownTrue.GetAlwaysTrueExpr().ProtoReflect().SetUnknown([]byte{0xa0, 0x06, 1})
	unknownUnary := notSource(opaque)
	unknownUnary.GetUnaryExpr().ProtoReflect().SetUnknown([]byte{0xa0, 0x06, 1})
	unknownBinary := binarySource(planpb.BinaryExpr_LogicalAnd, trueSource(), opaque)
	unknownBinary.GetBinaryExpr().ProtoReflect().SetUnknown([]byte{0xa0, 0x06, 1})
	nested := proto.Clone(opaque).(*planpb.Expr)
	nested.GetNullExpr().ColumnInfo.NestedPath = []string{"x"}
	for _, tc := range []struct {
		name   string
		source *planpb.Expr
		kind   translation.FailureKind
	}{
		{"unknown child", binarySource(planpb.BinaryExpr_LogicalAnd, opaque, notSource(unknown)), translation.UnsupportedExpression},
		{"excluded child", notSource(nested), translation.UnsupportedExpression},
		{"defect after limitation", binarySource(planpb.BinaryExpr_LogicalOr, unknown, malformed), translation.UnsupportedExpression},
		{"defect before limitation", binarySource(planpb.BinaryExpr_LogicalOr, malformed, unknown), translation.UnsafeSource},
		{"late defect", binarySource(planpb.BinaryExpr_LogicalAnd, opaque, notSource(malformed)), translation.UnsafeSource},
		{"missing true", &planpb.Expr{Expr: &planpb.Expr_AlwaysTrueExpr{}}, translation.UnsafeSource},
		{"nil true wrapper", &planpb.Expr{Expr: (*planpb.Expr_AlwaysTrueExpr)(nil)}, translation.UnsafeSource},
		{"nil unary wrapper", &planpb.Expr{Expr: (*planpb.Expr_UnaryExpr)(nil)}, translation.UnsafeSource},
		{"nil binary wrapper", &planpb.Expr{Expr: (*planpb.Expr_BinaryExpr)(nil)}, translation.UnsafeSource},
		{"missing unary child", notSource(nil), translation.UnsafeSource},
		{"missing binary child", binarySource(planpb.BinaryExpr_LogicalAnd, opaque, nil), translation.UnsafeSource},
		{"invalid unary operator", badOp, translation.UnsafeSource},
		{"invalid binary operator", badBinary, translation.UnsafeSource},
		{"future operator", future, translation.UnsupportedExpression},
		{"unknown true payload", unknownTrue, translation.UnsupportedExpression},
		{"unknown unary payload", unknownUnary, translation.UnsupportedExpression},
		{"unknown binary payload", unknownBinary, translation.UnsupportedExpression},
	} {
		t.Run(tc.name, func(t *testing.T) {
			imported, err := translator.Import(context.Background(), tc.source)
			require.Nil(t, imported)
			assertFailure(t, err, tc.kind, translation.StageImport)
		})
	}
}

func TestLogicalCancellationDuringTraversal(t *testing.T) {
	translator, opaque, _ := opaqueFixture(t, schemapb.DataType_VarChar, true)
	source := binarySource(planpb.BinaryExpr_LogicalAnd, trueSource(), notSource(opaque))
	imported := importSource(t, translator, source)
	candidate, err := translator.Export(context.Background(), imported, imported.Root())
	require.NoError(t, err)
	// Exhaust every context-check position until each stage succeeds, covering late descendants and final return.
	for name, stage := range map[string]func(context.Context) error{
		"import": func(ctx context.Context) error {
			result, err := translator.Import(ctx, source)
			if err != nil {
				require.Nil(t, result)
			}
			return err
		},
		"export": func(ctx context.Context) error {
			result, err := translator.Export(ctx, imported, imported.Root())
			if err != nil {
				require.Nil(t, result)
			}
			return err
		},
	} {
		t.Run(name, func(t *testing.T) {
			for checks := 1; checks < 32; checks++ {
				err := stage(&cancellationContext{Context: context.Background(), remaining: checks})
				if err == nil {
					require.Greater(t, checks, 4)
					return
				}
				require.ErrorIs(t, err, context.Canceled)
			}
			t.Fatal("stage did not reach a complete result")
		})
	}
	require.True(t, proto.Equal(expectedExport(source), candidate))
}
