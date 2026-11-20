package translation_test

import (
	"context"
	"testing"

	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/foundation"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	adapterruntime "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/runtime"
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
	_, imported := importSource(t, translator, source)
	root := imported.Root().(*expressions.LogicalExpression)
	require.Equal(t, foundation.And, root.Operator())
	left := root.ChildAt(0).(*expressions.LogicalExpression)
	right := root.ChildAt(1).(*expressions.LogicalExpression)
	require.Equal(t, foundation.Or, left.Operator())
	require.Equal(t, foundation.And, right.Operator())
	require.Equal(t, foundation.Not, left.ChildAt(1).(*expressions.LogicalExpression).Operator())
	require.Equal(t, foundation.Not, right.ChildAt(1).(*expressions.LogicalExpression).ChildAt(0).(*expressions.LogicalExpression).Operator())
	require.False(t, left.ChildAt(0).(*expressions.OpaqueExpression).ID().EqualsTo(right.ChildAt(0).(*expressions.OpaqueExpression).ID()))
	// Equal-looking opaque leaves still cannot substitute for another source occurrence.
	wrongOccurrence := logicalGraph(t, foundation.And,
		logicalGraph(t, foundation.Or, right.ChildAt(0), left.ChildAt(1)), right)
	rejected, err := translator.Export(ctx, imported, wrongOccurrence)
	require.Nil(t, rejected)
	assertFailure(t, err, translation.TranslationDefect, translation.StageExport)
	// Rebuild the root and share an equivalent native TRUE leaf at positions with different metadata.
	sharedLeaf := left.ChildAt(1).ChildAt(0)
	rebuilt := logicalGraph(t, foundation.And, left,
		logicalGraph(t, foundation.And, right.ChildAt(0), logicalGraph(t, foundation.Not, logicalGraph(t, foundation.Not, sharedLeaf))))
	candidate, err := translator.Export(ctx, imported, rebuilt)
	require.NoError(t, err)
	require.NoError(t, translator.ValidateCandidate(ctx, imported, candidate))
	require.True(t, proto.Equal(source, candidate))
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
	require.True(t, proto.Equal(source, second))
	require.Empty(t, opaqueRight.GetNullExpr().ColumnInfo.NestedPath)
	assertFailure(t, translator.ValidateCandidate(ctx, imported, candidate), translation.TranslationDefect, translation.StageCandidateValidation)
	// Root constants use the same native routes, preserving FALSE as NOT TRUE.
	for _, source := range []*planpb.Expr{trueSource(), notSource(trueSource()), binarySource(planpb.BinaryExpr_LogicalAnd, sharedTrue, sharedTrue)} {
		_, imported := importSource(t, translator, source)
		candidate, err := translator.Export(ctx, imported, imported.Root())
		require.NoError(t, err)
		require.True(t, proto.Equal(source, candidate))
		require.NoError(t, translator.ValidateCandidate(ctx, imported, candidate))
	}
}

func TestLogicalAdmissionIsWholeTree(t *testing.T) {
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
		{"unknown child", binarySource(planpb.BinaryExpr_LogicalAnd, opaque, notSource(unknown)), translation.UnsupportedSource},
		{"excluded child", notSource(nested), translation.UnsupportedSource},
		{"defect after limitation", binarySource(planpb.BinaryExpr_LogicalOr, unknown, malformed), translation.UnsafeSource},
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
		{"future operator", future, translation.UnsupportedSource},
		{"unknown true payload", unknownTrue, translation.UnsupportedSource},
		{"unknown unary payload", unknownUnary, translation.UnsupportedSource},
		{"unknown binary payload", unknownBinary, translation.UnsupportedSource},
	} {
		t.Run(tc.name, func(t *testing.T) {
			admitted, err := translator.Admit(context.Background(), tc.source)
			require.Nil(t, admitted)
			assertFailure(t, err, tc.kind, translation.StageAdmission)
		})
	}
}

func TestLogicalGraphAndCandidateChanges(t *testing.T) {
	ctx := context.Background()
	translator, opaque, _ := opaqueFixture(t, schemapb.DataType_Bool, false)
	source := binarySource(planpb.BinaryExpr_LogicalAnd, opaque, notSource(trueSource()))
	_, imported := importSource(t, translator, source)
	root := imported.Root().(*expressions.LogicalExpression)
	rt, err := adapterruntime.SharedRuntime()
	require.NoError(t, err)
	falseValue, err := rt.ValueFactory().NewBool(rt.Types().BoolNonNull, false)
	require.NoError(t, err)
	falseLiteral, err := expressions.NewLiteralExpression(falseValue)
	require.NoError(t, err)
	nullable, err := expressions.NewOpaqueExpression(root.ChildAt(0).(*expressions.OpaqueExpression).ID(), rt.Types().BoolMayNull, "changed")
	require.NoError(t, err)
	for _, graph := range []expressions.Expression{
		logicalGraph(t, foundation.Or, root.ChildAt(0), root.ChildAt(1)),
		logicalGraph(t, foundation.And, root.ChildAt(1), root.ChildAt(0)),
		logicalGraph(t, foundation.And, root.ChildAt(0), root.ChildAt(1), root.ChildAt(1)),
		logicalGraph(t, foundation.And, root.ChildAt(0), falseLiteral),
		logicalGraph(t, foundation.And, root.ChildAt(0), logicalGraph(t, foundation.Not, falseLiteral)),
		logicalGraph(t, foundation.And, nullable, root.ChildAt(1)),
	} {
		candidate, err := translator.Export(ctx, imported, graph)
		require.Nil(t, candidate)
		assertFailure(t, err, translation.TranslationDefect, translation.StageExport)
	}
	for _, mutate := range []func(*planpb.Expr){
		func(e *planpb.Expr) { e.GetBinaryExpr().Op = planpb.BinaryExpr_LogicalOr },
		func(e *planpb.Expr) { e.GetBinaryExpr().Right = trueSource() },
		func(e *planpb.Expr) { e.GetBinaryExpr().Right.GetUnaryExpr().Child.IsTemplate = true },
		func(e *planpb.Expr) { e.GetBinaryExpr().Left.GetNullExpr().Op = planpb.NullExpr_IsNotNull },
		func(e *planpb.Expr) { e.GetBinaryExpr().Right.GetUnaryExpr().Child = nil },
		func(e *planpb.Expr) {
			e.GetBinaryExpr().Right.GetUnaryExpr().ProtoReflect().SetUnknown([]byte{0xa0, 0x06, 1})
		},
	} {
		candidate, err := translator.Export(ctx, imported, root)
		require.NoError(t, err)
		mutate(candidate)
		assertFailure(t, translator.ValidateCandidate(ctx, imported, candidate), translation.TranslationDefect, translation.StageCandidateValidation)
	}
}

func TestLogicalCancellationDuringTraversal(t *testing.T) {
	translator, opaque, _ := opaqueFixture(t, schemapb.DataType_VarChar, true)
	source := binarySource(planpb.BinaryExpr_LogicalAnd, trueSource(), notSource(opaque))
	admitted, imported := importSource(t, translator, source)
	candidate, err := translator.Export(context.Background(), imported, imported.Root())
	require.NoError(t, err)
	// Exhaust every context-check position until each stage succeeds, covering late descendants and final return.
	for name, stage := range map[string]func(context.Context) error{
		"admit": func(ctx context.Context) error {
			result, err := translator.Admit(ctx, source)
			if err != nil {
				require.Nil(t, result)
			}
			return err
		},
		"import": func(ctx context.Context) error {
			result, err := translator.Import(ctx, admitted)
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
		"candidate": func(ctx context.Context) error { return translator.ValidateCandidate(ctx, imported, candidate) },
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
	require.True(t, proto.Equal(source, candidate))
}
