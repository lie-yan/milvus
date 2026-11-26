package translation_test

import (
	"context"
	"testing"

	"github.com/lie-yan/predicate-optimization/expressions"
	"github.com/lie-yan/predicate-optimization/foundation"
	shared "github.com/lie-yan/predicate-optimization/functions"
	functioninterfaces "github.com/lie-yan/predicate-optimization/functions/interfaces"
	valueinterfaces "github.com/lie-yan/predicate-optimization/values/interfaces"
	"github.com/stretchr/testify/require"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	adapterruntime "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/runtime"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/schema"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/translation"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// Any evaluation during setup or translation fails the test immediately.
type noEvaluationContext struct {
	functioninterfaces.FunctionContext
}

func (*noEvaluationContext) Evaluate(functioninterfaces.FunctionHandle, *shared.FunctionSignature, []valueinterfaces.Value) (valueinterfaces.Value, error) {
	panic("translation must not evaluate")
}

type guardedRuntime struct {
	translation.Runtime
	provider functioninterfaces.FunctionContext
}

func (r *guardedRuntime) FunctionContext() functioninterfaces.FunctionContext { return r.provider }

func opaqueFixture(t *testing.T, typ schemapb.DataType, nullable bool) (*translation.Translator, *planpb.Expr, *schemapb.FieldSchema) {
	t.Helper()
	field := &schemapb.FieldSchema{FieldID: 101, Name: "field", DataType: typ, Nullable: nullable}
	helper, err := typeutil.CreateSchemaHelper(&schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{field}})
	require.NoError(t, err)
	view, err := schema.NewView(helper, []*schemapb.FieldSchema{field})
	require.NoError(t, err)
	runtime, err := adapterruntime.SharedRuntime()
	require.NoError(t, err)
	guarded := &guardedRuntime{Runtime: runtime, provider: &noEvaluationContext{runtime.FunctionContext()}}
	scalar := runtime.Types()
	translator, err := translation.New(guarded, view, scalar)
	require.NoError(t, err)
	source := &planpb.Expr{Expr: &planpb.Expr_NullExpr{NullExpr: &planpb.NullExpr{
		Op:         planpb.NullExpr_IsNull,
		ColumnInfo: &planpb.ColumnInfo{FieldId: field.FieldID, DataType: typ, Nullable: nullable},
	}}}
	return translator, source, field
}

func importSource(t *testing.T, translator *translation.Translator, source *planpb.Expr) *translation.ImportResult {
	t.Helper()
	imported, err := translator.Import(context.Background(), source)
	require.NoError(t, err)
	return imported
}

func assertFailure(t *testing.T, err error, kind translation.FailureKind, stage translation.Stage) {
	t.Helper()
	var failure *translation.Failure
	require.ErrorAs(t, err, &failure)
	require.Equal(t, kind, failure.Kind())
	require.Equal(t, stage, failure.Stage())
}

// Deterministic cancellation at stage entry, handler entry or the final return check.
type cancellationContext struct {
	context.Context
	remaining int
}

func (c *cancellationContext) Err() error {
	c.remaining--
	if c.remaining <= 0 {
		return context.Canceled
	}
	return nil
}

func trueSource() *planpb.Expr {
	return &planpb.Expr{Expr: &planpb.Expr_AlwaysTrueExpr{AlwaysTrueExpr: &planpb.AlwaysTrueExpr{}}}
}

func notSource(child *planpb.Expr) *planpb.Expr {
	return &planpb.Expr{Expr: &planpb.Expr_UnaryExpr{UnaryExpr: &planpb.UnaryExpr{Op: planpb.UnaryExpr_Not, Child: child}}}
}

func binarySource(op planpb.BinaryExpr_BinaryOp, left, right *planpb.Expr) *planpb.Expr {
	return &planpb.Expr{Expr: &planpb.Expr_BinaryExpr{BinaryExpr: &planpb.BinaryExpr{Op: op, Left: left, Right: right}}}
}

func logicalGraph(t *testing.T, op foundation.LogicalOperator, children ...expressions.Expression) expressions.Expression {
	t.Helper()
	graph, err := expressions.NewLogicalExpression(op, children)
	require.NoError(t, err)
	return graph
}

// Checks graph operators independently of protobuf equality, including repeated source positions.

func unarySource(column *planpb.ColumnInfo, op planpb.OpType, value *planpb.GenericValue) *planpb.Expr {
	return &planpb.Expr{Expr: &planpb.Expr_UnaryRangeExpr{UnaryRangeExpr: &planpb.UnaryRangeExpr{ColumnInfo: column, Op: op, Value: value}}}
}

func integerSource(value int64) *planpb.GenericValue {
	return &planpb.GenericValue{Val: &planpb.GenericValue_Int64Val{Int64Val: value}}
}

func rebuiltCall(t *testing.T, original *expressions.CallExpression, args ...expressions.Expression) *expressions.CallExpression {
	t.Helper()
	graph, err := expressions.NewCallExpression(original.Context(), original.Handle(), args)
	require.NoError(t, err)
	return graph
}
