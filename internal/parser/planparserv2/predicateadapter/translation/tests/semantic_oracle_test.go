package translation_test

import (
	"cmp"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

// Independent row semantics for the admitted scalar test domain, not a backend emulator.
func evaluatePredicate(t testing.TB, expr *planpb.Expr, row map[int64]*planpb.GenericValue) bool {
	t.Helper()
	column := func(info *planpb.ColumnInfo) *planpb.GenericValue {
		value, present := row[info.FieldId]
		require.True(t, present, "missing field %d", info.FieldId)
		return value
	}
	switch node := expr.Expr.(type) {
	case *planpb.Expr_AlwaysTrueExpr:
		return true
	case *planpb.Expr_UnaryExpr:
		require.Equal(t, planpb.UnaryExpr_Not, node.UnaryExpr.Op)
		return !evaluatePredicate(t, node.UnaryExpr.Child, row)
	case *planpb.Expr_BinaryExpr:
		left := evaluatePredicate(t, node.BinaryExpr.Left, row)
		right := evaluatePredicate(t, node.BinaryExpr.Right, row)
		switch node.BinaryExpr.Op {
		case planpb.BinaryExpr_LogicalAnd:
			return left && right
		case planpb.BinaryExpr_LogicalOr:
			return left || right
		}
	case *planpb.Expr_UnaryRangeExpr:
		return compareValues(t, node.UnaryRangeExpr.Op, column(node.UnaryRangeExpr.ColumnInfo), node.UnaryRangeExpr.Value)
	case *planpb.Expr_CompareExpr:
		return compareValues(t, node.CompareExpr.Op, column(node.CompareExpr.LeftColumnInfo), column(node.CompareExpr.RightColumnInfo))
	case *planpb.Expr_BinaryRangeExpr:
		r := node.BinaryRangeExpr
		lower, upper := planpb.OpType_GreaterThan, planpb.OpType_LessThan
		if r.LowerInclusive {
			lower = planpb.OpType_GreaterEqual
		}
		if r.UpperInclusive {
			upper = planpb.OpType_LessEqual
		}
		value := column(r.ColumnInfo)
		return compareValues(t, lower, value, r.LowerValue) && compareValues(t, upper, value, r.UpperValue)
	case *planpb.Expr_TermExpr:
		value := column(node.TermExpr.ColumnInfo)
		for _, candidate := range node.TermExpr.Values {
			if compareValues(t, planpb.OpType_Equal, value, candidate) {
				return true
			}
		}
		return false
	case *planpb.Expr_NullExpr:
		isNull := column(node.NullExpr.ColumnInfo) == nil
		switch node.NullExpr.Op {
		case planpb.NullExpr_IsNull:
			return isNull
		case planpb.NullExpr_IsNotNull:
			return !isNull
		}
	}
	t.Fatalf("unsupported oracle expression: %v", expr)
	return false
}

func compareValues(t testing.TB, op planpb.OpType, left, right *planpb.GenericValue) bool {
	t.Helper()
	require.NotNil(t, left, "native comparisons require nonnull values")
	require.NotNil(t, right)
	require.IsType(t, left.Val, right.Val)
	switch value := left.Val.(type) {
	case *planpb.GenericValue_Int64Val:
		return orderedComparison(t, op, value.Int64Val, right.GetInt64Val())
	case *planpb.GenericValue_FloatVal:
		return orderedComparison(t, op, value.FloatVal, right.GetFloatVal())
	case *planpb.GenericValue_StringVal:
		return orderedComparison(t, op, value.StringVal, right.GetStringVal())
	case *planpb.GenericValue_BoolVal:
		switch op {
		case planpb.OpType_Equal:
			return value.BoolVal == right.GetBoolVal()
		case planpb.OpType_NotEqual:
			return value.BoolVal != right.GetBoolVal()
		}
	}
	t.Fatalf("unsupported oracle comparison: %v %v %v", left, op, right)
	return false
}

func orderedComparison[T cmp.Ordered](t testing.TB, op planpb.OpType, left, right T) bool {
	t.Helper()
	switch op {
	case planpb.OpType_Equal:
		return left == right
	case planpb.OpType_NotEqual:
		return left != right
	case planpb.OpType_LessThan:
		return left < right
	case planpb.OpType_LessEqual:
		return left <= right
	case planpb.OpType_GreaterThan:
		return left > right
	case planpb.OpType_GreaterEqual:
		return left >= right
	}
	t.Fatalf("unsupported oracle operator: %v", op)
	return false
}
