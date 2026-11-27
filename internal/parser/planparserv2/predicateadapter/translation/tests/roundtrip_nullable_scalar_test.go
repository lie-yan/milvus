package translation_test

import (
	"context"
	"strings"
	"testing"

	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/foundation"
	"github.com/lie-yan/predicate-optimization/types"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/translation"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/values"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

func TestVarCharEqualityMappings(t *testing.T) {
	for _, nullable := range []bool{false, true} {
		translator, fields := compareFixture(t, schemapb.DataType_VarChar, schemapb.DataType_VarChar, nullable)
		column := fields.GetCompareExpr().RightColumnInfo
		for _, op := range []planpb.OpType{planpb.OpType_Equal, planpb.OpType_NotEqual} {
			for _, payload := range []string{"", "a\x00猫", strings.Repeat("é", 50)} {
				source := unarySource(column, op, stringSource(payload))
				source.IsTemplate = true
				source.GetUnaryRangeExpr().TemplateVariableName = "bound"
				original := proto.Clone(source)
				imported := importSource(t, translator, source)
				call := imported.Root().(*expressions.CallExpression)
				reference, literal := call.ChildAt(0), call.ChildAt(1)
				expectedNullability := foundation.NonNull
				if nullable {
					expectedNullability = foundation.MayNull
				}
				require.Equal(t, expectedNullability, reference.Type().Nullability())
				require.Equal(t, expectedNullability, call.Type().Nullability())
				require.Equal(t, foundation.NonNull, literal.Type().Nullability())
				bound, bounded := reference.Type().(*types.VarCharType).MaxBytes()
				require.True(t, bounded)
				require.Equal(t, uint64(100), bound)
				require.True(t, reference.Type().EqualsToIgnoringNullability(literal.Type()))
				for _, graph := range []*expressions.CallExpression{call, rebuiltCall(t, call, literal, reference)} {
					output, err := translator.Export(context.Background(), imported, graph)
					require.NoError(t, err)
					require.True(t, proto.Equal(expectedExport(source), output))
					output.GetUnaryRangeExpr().ColumnInfo.Nullable = !nullable
					output.GetUnaryRangeExpr().Value.Val.(*planpb.GenericValue_StringVal).StringVal = "changed"
					require.True(t, proto.Equal(original, source))
				}
				// A rewritten comparison may contain two references.
				output, err := translator.Export(context.Background(), imported, rebuiltCall(t, call, reference, reference))
				require.NoError(t, err)
				require.True(t, proto.Equal(column, output.GetCompareExpr().LeftColumnInfo))
				require.True(t, proto.Equal(column, output.GetCompareExpr().RightColumnInfo))
			}
		}
		for _, source := range []*planpb.Expr{
			unarySource(column, planpb.OpType_Equal, stringSource(strings.Repeat("é", 51))),
			unarySource(column, planpb.OpType_LessThan, stringSource("a")),
		} {
			imported, err := translator.Import(context.Background(), source)
			require.Nil(t, imported)
			assertFailure(t, err, translation.UnsupportedExpression, translation.StageImport)
		}
	}
}

func TestNullableMembershipAndLogicalExport(t *testing.T) {
	for _, typ := range []schemapb.DataType{schemapb.DataType_Int64, schemapb.DataType_VarChar} {
		translator, fields := compareFixture(t, typ, typ, true)
		candidates := []*planpb.GenericValue{integerSource(9007199254740993), integerSource(-7), integerSource(9007199254740993)}
		if typ == schemapb.DataType_VarChar {
			candidates = []*planpb.GenericValue{stringSource("猫"), stringSource(""), stringSource("猫")}
		}
		leaf := termSource(fields.GetCompareExpr().RightColumnInfo, candidates...)
		source := binarySource(planpb.BinaryExpr_LogicalOr, leaf, notSource(leaf))
		original := proto.Clone(source)
		imported := importSource(t, translator, source)
		term := imported.Root().ChildAt(0).(*expressions.SpecialFormExpression)
		require.Equal(t, foundation.MayNull, term.Type().Nullability())
		require.Equal(t, foundation.MayNull, term.ChildAt(0).Type().Nullability())
		for _, literal := range term.Children()[1:] {
			require.Equal(t, foundation.NonNull, literal.Type().Nullability())
		}
		nullValue, err := values.Factory.NewNull(term.ChildAt(0).Type())
		require.NoError(t, err)
		nullLiteral, err := expressions.NewLiteralExpression(nullValue)
		require.NoError(t, err)
		withNull, err := expressions.NewSpecialFormExpression(foundation.SpecialFormIn, term.Type(), []expressions.Expression{term.ChildAt(0), nullLiteral})
		require.NoError(t, err)
		rejected, err := translator.Export(context.Background(), imported, withNull)
		require.Nil(t, rejected)
		assertFailure(t, err, translation.UnsupportedExpression, translation.StageExport)
		// Changed n-ary graphs share a nullable leaf; export must lower each occurrence independently.
		for _, op := range []foundation.LogicalOperator{foundation.And, foundation.Or} {
			negated := logicalGraph(t, foundation.Not, term)
			graph := logicalGraph(t, op, term, negated, term)
			require.Equal(t, foundation.MayNull, graph.Type().Nullability())
			output, err := translator.Export(context.Background(), imported, graph)
			require.NoError(t, err)
			nativeOp := planpb.BinaryExpr_LogicalAnd
			if op == foundation.Or {
				nativeOp = planpb.BinaryExpr_LogicalOr
			}
			expected := binarySource(nativeOp, leaf, binarySource(nativeOp, notSource(leaf), leaf))
			require.True(t, proto.Equal(expectedExport(expected), output))
			for _, rowValue := range []*planpb.GenericValue{nil, candidates[0]} {
				row := map[int64]*planpb.GenericValue{102: rowValue}
				require.Equal(t, nullablePredicate(t, expected, row), nullablePredicate(t, output, row))
				if rowValue == nil {
					require.Equal(t, -1, nullablePredicate(t, output, row))
				}
			}
			output.GetBinaryExpr().Left.GetTermExpr().Values[0].Val = integerSource(42).Val
			output.GetBinaryExpr().Left.GetTermExpr().ColumnInfo.Nullable = false
			again, err := translator.Export(context.Background(), imported, graph)
			require.NoError(t, err)
			require.True(t, proto.Equal(expectedExport(expected), again))
			require.True(t, proto.Equal(original, source))
		}
	}
}

// Independent three-valued row oracle: -1 UNKNOWN, 0 FALSE, 1 TRUE; no native execution claim.
func nullablePredicate(t *testing.T, expr *planpb.Expr, row map[int64]*planpb.GenericValue) int {
	t.Helper()
	switch n := expr.Expr.(type) {
	case *planpb.Expr_UnaryExpr:
		child := nullablePredicate(t, n.UnaryExpr.Child, row)
		if child < 0 {
			return child
		}
		return 1 - child
	case *planpb.Expr_BinaryExpr:
		left, right := nullablePredicate(t, n.BinaryExpr.Left, row), nullablePredicate(t, n.BinaryExpr.Right, row)
		if n.BinaryExpr.Op == planpb.BinaryExpr_LogicalAnd {
			if left == 0 || right == 0 {
				return 0
			}
			if left < 0 || right < 0 {
				return -1
			}
			return 1
		}
		if left == 1 || right == 1 {
			return 1
		}
		if left < 0 || right < 0 {
			return -1
		}
		return 0
	case *planpb.Expr_TermExpr:
		if row[n.TermExpr.ColumnInfo.FieldId] == nil {
			return -1
		}
	case *planpb.Expr_UnaryRangeExpr:
		if row[n.UnaryRangeExpr.ColumnInfo.FieldId] == nil {
			return -1
		}
	}
	if evaluatePredicate(t, expr, row) {
		return 1
	}
	return 0
}

func TestNullableLogicalTruthTable(t *testing.T) {
	translator, fields := compareFixture(t, schemapb.DataType_Int64, schemapb.DataType_Int64, true)
	left := termSource(fields.GetCompareExpr().LeftColumnInfo, integerSource(1))
	right := notSource(termSource(fields.GetCompareExpr().RightColumnInfo, integerSource(1)))
	for _, op := range []planpb.BinaryExpr_BinaryOp{planpb.BinaryExpr_LogicalAnd, planpb.BinaryExpr_LogicalOr} {
		source := binarySource(op, left, right)
		imported := importSource(t, translator, source)
		require.Equal(t, foundation.MayNull, imported.Root().Type().Nullability())
		output, err := translator.Export(context.Background(), imported, imported.Root())
		require.NoError(t, err)
		// Columns select FALSE/TRUE on the left and UNKNOWN/FALSE/TRUE on the right.
		expected := [2][3]int{{0, 0, 0}, {-1, 0, 1}}
		if op == planpb.BinaryExpr_LogicalOr {
			expected = [2][3]int{{-1, 0, 1}, {1, 1, 1}}
		}
		for i, l := range []*planpb.GenericValue{integerSource(0), integerSource(1)} {
			for j, r := range []*planpb.GenericValue{nil, integerSource(1), integerSource(0)} {
				row := map[int64]*planpb.GenericValue{101: l, 102: r}
				require.Equal(t, expected[i][j], nullablePredicate(t, source, row))
				require.Equal(t, expected[i][j], nullablePredicate(t, output, row))
			}
		}
	}
}
