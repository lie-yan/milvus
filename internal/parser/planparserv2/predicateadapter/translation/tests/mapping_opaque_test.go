package translation_test

import (
	"context"
	"errors"
	"testing"

	"github.com/lie-yan/predicate-optimization/expressions"
	expressioninterfaces "github.com/lie-yan/predicate-optimization/expressions/interfaces"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	adapterexpressions "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/expressions"
	adapterruntime "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/runtime"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/translation"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// External identity representation, deliberately not Go-comparable.
type opaquePeer struct{ parts []string }

func (id opaquePeer) Key() string    { return id.parts[0] }
func (id opaquePeer) String() string { return "rebuilt region" }
func (id opaquePeer) EqualsTo(other expressioninterfaces.OpaqueID) bool {
	return id.Key() == other.Key()
}

func TestOpaqueScalarNullRoundTrip(t *testing.T) {
	runtime, err := adapterruntime.SharedRuntime()
	require.NoError(t, err)
	for _, typ := range []schemapb.DataType{schemapb.DataType_Bool, schemapb.DataType_Int8, schemapb.DataType_Int16, schemapb.DataType_Int32, schemapb.DataType_Int64, schemapb.DataType_Float, schemapb.DataType_Double, schemapb.DataType_VarChar} {
		for _, nullable := range []bool{false, true} {
			for _, op := range []planpb.NullExpr_NullOp{planpb.NullExpr_IsNull, planpb.NullExpr_IsNotNull} {
				t.Run(typ.String()+"/"+op.String()+map[bool]string{false: "/required", true: "/nullable"}[nullable], func(t *testing.T) {
					translator, source, _ := opaqueFixture(t, typ, nullable)
					source.GetNullExpr().Op = op
					// Preserve envelope metadata as received at the completed parser boundary.
					source.IsTemplate = true
					_, imported := importSource(t, translator, source)
					leaf := imported.Root().(*expressions.OpaqueExpression)
					require.Same(t, runtime.Types().BoolNonNull, leaf.Type())
					rebuilt, err := expressions.NewOpaqueExpression(opaquePeer{[]string{leaf.ID().Key()}}, leaf.Type(), "different spelling")
					require.NoError(t, err)
					candidate, err := translator.Export(context.Background(), imported, rebuilt)
					require.NoError(t, err)
					require.NoError(t, translator.ValidateCandidate(context.Background(), imported, candidate))
					require.True(t, proto.Equal(source, candidate))
				})
			}
		}
	}
}

func TestOpaqueAdmissionFailures(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*planpb.Expr) *planpb.Expr
		kind   translation.FailureKind
	}{
		{"nil root", func(*planpb.Expr) *planpb.Expr { return nil }, translation.UnsafeSource},
		{"missing arm", func(*planpb.Expr) *planpb.Expr { return &planpb.Expr{} }, translation.UnsafeSource},
		{"typed nil arm", func(e *planpb.Expr) *planpb.Expr { e.Expr = (*planpb.Expr_NullExpr)(nil); return e }, translation.UnsafeSource},
		{"missing message", func(e *planpb.Expr) *planpb.Expr { e.Expr = &planpb.Expr_NullExpr{}; return e }, translation.UnsafeSource},
		{"missing column", func(e *planpb.Expr) *planpb.Expr { e.GetNullExpr().ColumnInfo = nil; return e }, translation.UnsafeSource},
		{"zero operator", func(e *planpb.Expr) *planpb.Expr { e.GetNullExpr().Op = planpb.NullExpr_Invalid; return e }, translation.UnsafeSource},
		{"future operator", func(e *planpb.Expr) *planpb.Expr { e.GetNullExpr().Op = 99; return e }, translation.UnsupportedSource},
		{"missing schema field", func(e *planpb.Expr) *planpb.Expr { e.GetNullExpr().ColumnInfo.FieldId = 999; return e }, translation.UnsafeSource},
		{"changed nullability", func(e *planpb.Expr) *planpb.Expr { e.GetNullExpr().ColumnInfo.Nullable = false; return e }, translation.UnsafeSource},
		{"changed datatype", func(e *planpb.Expr) *planpb.Expr {
			e.GetNullExpr().ColumnInfo.DataType = schemapb.DataType_Int64
			return e
		}, translation.UnsafeSource},
		{"nested scope", func(e *planpb.Expr) *planpb.Expr { e.GetNullExpr().ColumnInfo.NestedPath = []string{"x"}; return e }, translation.UnsupportedSource},
		{"element scope", func(e *planpb.Expr) *planpb.Expr { e.GetNullExpr().ColumnInfo.IsElementLevel = true; return e }, translation.UnsupportedSource},
		{"unknown call type", func(*planpb.Expr) *planpb.Expr {
			return &planpb.Expr{Expr: &planpb.Expr_CallExpr{CallExpr: &planpb.CallExpr{FunctionName: "unknown"}}}
		}, translation.UnsupportedSource},
	} {
		t.Run(tc.name, func(t *testing.T) {
			translator, source, _ := opaqueFixture(t, schemapb.DataType_VarChar, true)
			admitted, err := translator.Admit(context.Background(), tc.change(source))
			require.Nil(t, admitted)
			assertFailure(t, err, tc.kind, translation.StageAdmission)
			if tc.kind == translation.UnsafeSource {
				require.ErrorIs(t, err, merr.ErrServiceInternal)
			}
		})
	}
	for _, layer := range []string{"envelope", "message", "column"} {
		t.Run("unknown "+layer, func(t *testing.T) {
			translator, source, _ := opaqueFixture(t, schemapb.DataType_Bool, false)
			var message proto.Message = source
			if layer == "message" {
				message = source.GetNullExpr()
			}
			if layer == "column" {
				message = source.GetNullExpr().ColumnInfo
			}
			message.ProtoReflect().SetUnknown([]byte{0xa0, 0x06, 1})
			admitted, err := translator.Admit(context.Background(), source)
			require.Nil(t, admitted)
			assertFailure(t, err, translation.UnsupportedSource, translation.StageAdmission)
		})
	}
	translator, source, _ := opaqueFixture(t, schemapb.DataType_JSON, true)
	admitted, err := translator.Admit(context.Background(), source)
	require.Nil(t, admitted)
	assertFailure(t, err, translation.UnsupportedSource, translation.StageAdmission)
}

func TestOpaqueGraphOwnershipAndCandidateValidation(t *testing.T) {
	translator, source, field := opaqueFixture(t, schemapb.DataType_VarChar, true)
	admitted, imported := importSource(t, translator, source)
	otherImport, err := translator.Import(context.Background(), admitted)
	require.NoError(t, err)
	root := imported.Root().(*expressions.OpaqueExpression)
	otherRoot := otherImport.Root().(*expressions.OpaqueExpression)
	require.False(t, root.ID().EqualsTo(otherRoot.ID()))
	runtime, err := adapterruntime.SharedRuntime()
	require.NoError(t, err)
	changedType, err := expressions.NewOpaqueExpression(root.ID(), runtime.Types().BoolMayNull, "")
	require.NoError(t, err)
	unregistered, err := expressions.NewOpaqueExpression(opaquePeer{[]string{root.ID().Key() + "0"}}, root.Type(), "")
	require.NoError(t, err)
	foreignID, err := expressions.NewOpaqueExpression(adapterexpressions.NewOpaqueID(1, 999), root.Type(), "")
	require.NoError(t, err)
	value, err := runtime.ValueFactory().NewBool(runtime.Types().BoolNonNull, true)
	require.NoError(t, err)
	literal, err := expressions.NewLiteralExpression(value)
	require.NoError(t, err)
	for _, graph := range []expressions.Expression{otherRoot, changedType, unregistered, foreignID, literal, (*expressions.OpaqueExpression)(nil)} {
		output, err := translator.Export(context.Background(), imported, graph)
		require.Nil(t, output)
		assertFailure(t, err, translation.TranslationDefect, translation.StageExport)
	}
	for _, mutate := range []func(*planpb.Expr){
		func(e *planpb.Expr) { e.IsTemplate = !e.IsTemplate },
		func(e *planpb.Expr) { e.GetNullExpr().Op = planpb.NullExpr_IsNotNull },
		func(e *planpb.Expr) { e.GetNullExpr().ColumnInfo.Nullable = false },
		func(e *planpb.Expr) { e.GetNullExpr().ColumnInfo.IsPrimaryKey = true },
		func(e *planpb.Expr) { e.GetNullExpr().ColumnInfo = nil },
		func(e *planpb.Expr) { e.GetNullExpr().ColumnInfo.ProtoReflect().SetUnknown([]byte{0xa0, 0x06, 1}) },
	} {
		output, err := translator.Export(context.Background(), imported, root)
		require.NoError(t, err)
		mutate(output)
		assertFailure(t, translator.ValidateCandidate(context.Background(), imported, output), translation.TranslationDefect, translation.StageCandidateValidation)
	}
	before := proto.Clone(source)
	first, err := translator.Export(context.Background(), imported, root)
	require.NoError(t, err)
	second, err := translator.Export(context.Background(), imported, root)
	require.NoError(t, err)
	require.NotSame(t, source, first)
	require.NotSame(t, source.Expr, first.Expr)
	require.NotSame(t, first.GetNullExpr(), second.GetNullExpr())
	require.NotSame(t, first.GetNullExpr().ColumnInfo, second.GetNullExpr().ColumnInfo)
	first.GetNullExpr().ColumnInfo.NestedPath = []string{"changed"}
	require.True(t, proto.Equal(before, source))
	require.True(t, proto.Equal(before, second))
	// After the final use of borrowed import state, source/schema mutations cannot affect output.
	source.GetNullExpr().Op = planpb.NullExpr_IsNotNull
	field.Nullable = false
	require.True(t, proto.Equal(before, second))
}

func TestOpaqueStageIsolationAndCancellation(t *testing.T) {
	translator, source, _ := opaqueFixture(t, schemapb.DataType_Int64, false)
	admitted, imported := importSource(t, translator, source)
	candidate, err := translator.Export(context.Background(), imported, imported.Root())
	require.NoError(t, err)
	foreign, _, _ := opaqueFixture(t, schemapb.DataType_Int64, false)
	result, err := foreign.Import(context.Background(), admitted)
	require.Nil(t, result)
	assertFailure(t, err, translation.TranslationDefect, translation.StageImport)
	output, err := foreign.Export(context.Background(), imported, imported.Root())
	require.Nil(t, output)
	assertFailure(t, err, translation.TranslationDefect, translation.StageExport)
	assertFailure(t, foreign.ValidateCandidate(context.Background(), imported, candidate), translation.TranslationDefect, translation.StageCandidateValidation)
	for _, checks := range []int{1, 2} {
		ctx := func() context.Context { return &cancellationContext{Context: context.Background(), remaining: checks} }
		a, err := translator.Admit(ctx(), source)
		require.Nil(t, a)
		require.ErrorIs(t, err, context.Canceled)
		i, err := translator.Import(ctx(), admitted)
		require.Nil(t, i)
		require.ErrorIs(t, err, context.Canceled)
		out, err := translator.Export(ctx(), imported, imported.Root())
		require.Nil(t, out)
		require.ErrorIs(t, err, context.Canceled)
		require.ErrorIs(t, translator.ValidateCandidate(ctx(), imported, candidate), context.Canceled)
	}
	// A single opaque export checks context at graph entry, graph handler, exporter, then final return.
	for _, checks := range []int{3, 4} {
		out, err := translator.Export(&cancellationContext{Context: context.Background(), remaining: checks}, imported, imported.Root())
		require.Nil(t, out)
		require.ErrorIs(t, err, context.Canceled)
		var failure *translation.Failure
		require.False(t, errors.As(err, &failure))
	}
	require.True(t, proto.Equal(source, candidate))
}
