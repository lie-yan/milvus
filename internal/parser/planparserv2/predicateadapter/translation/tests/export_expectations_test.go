package translation_test

import (
	"sort"

	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
)

// Converts historical source fixtures to expected executable form; semantic truth vectors are checked separately.
func expectedExport(source *planpb.Expr) *planpb.Expr {
	result := proto.Clone(source).(*planpb.Expr)
	switch n := result.Expr.(type) {
	case *planpb.Expr_NullExpr:
		return result // Opaque payloads retain exact metadata.
	case *planpb.Expr_BinaryExpr:
		n.BinaryExpr.Left = expectedExport(n.BinaryExpr.Left)
		n.BinaryExpr.Right = expectedExport(n.BinaryExpr.Right)
	case *planpb.Expr_UnaryExpr:
		n.UnaryExpr.Child = expectedExport(n.UnaryExpr.Child)
	case *planpb.Expr_UnaryRangeExpr:
		n.UnaryRangeExpr.TemplateVariableName = ""
	case *planpb.Expr_BinaryRangeExpr:
		r := n.BinaryRangeExpr
		lower, upper := planpb.OpType_GreaterThan, planpb.OpType_LessThan
		if r.LowerInclusive {
			lower = planpb.OpType_GreaterEqual
		}
		if r.UpperInclusive {
			upper = planpb.OpType_LessEqual
		}
		return binarySource(planpb.BinaryExpr_LogicalAnd, unarySource(r.ColumnInfo, lower, r.LowerValue), unarySource(proto.Clone(r.ColumnInfo).(*planpb.ColumnInfo), upper, r.UpperValue))
	case *planpb.Expr_TermExpr:
		n.TermExpr.TemplateVariableName = ""
		sort.SliceStable(n.TermExpr.Values, func(i, j int) bool {
			a, b := n.TermExpr.Values[i], n.TermExpr.Values[j]
			switch a.Val.(type) {
			case *planpb.GenericValue_BoolVal:
				return !a.GetBoolVal() && b.GetBoolVal()
			case *planpb.GenericValue_Int64Val:
				return a.GetInt64Val() < b.GetInt64Val()
			case *planpb.GenericValue_FloatVal:
				return a.GetFloatVal() < b.GetFloatVal()
			case *planpb.GenericValue_StringVal:
				return a.GetStringVal() < b.GetStringVal()
			default:
				panic("unsupported fixture scalar")
			}
		})
	}
	result.IsTemplate = false
	return result
}
