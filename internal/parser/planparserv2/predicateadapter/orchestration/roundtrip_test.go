package orchestration

import (
	"context"
	"testing"

	"github.com/lie-yan/predicate-optimization/expressions"
	shared "github.com/lie-yan/predicate-optimization/functions"
	functioninterfaces "github.com/lie-yan/predicate-optimization/functions/interfaces"
	"github.com/lie-yan/predicate-optimization/types"
	valueinterfaces "github.com/lie-yan/predicate-optimization/values/interfaces"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/functions"
	adapterruntime "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/runtime"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/schema"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/translation"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// Overrides only the selected stage, retaining real translation for preceding work.
type faultTranslator struct {
	pairedTranslator
	stage        Stage
	cancel       context.CancelFunc
	unclassified error
}

func (f *faultTranslator) Import(ctx context.Context, admitted *translation.AdmissionPlan) (*translation.ImportResult, error) {
	if f.stage == StageImport {
		if f.cancel != nil {
			f.cancel()
		}
		return f.pairedTranslator.Import(ctx, nil)
	}
	return f.pairedTranslator.Import(ctx, admitted)
}

func (f *faultTranslator) Export(ctx context.Context, imported *translation.ImportResult, graph expressions.Expression) (*planpb.Expr, error) {
	candidate, err := f.pairedTranslator.Export(ctx, imported, graph)
	if err != nil || f.stage != StageExport {
		return candidate, err
	}
	if f.cancel != nil {
		f.cancel()
	}
	if f.unclassified != nil {
		return candidate, f.unclassified
	}
	_, err = f.pairedTranslator.Export(ctx, nil, graph)
	return candidate, err // Even a candidate returned alongside an error must remain private.
}

func (f *faultTranslator) ValidateCandidate(ctx context.Context, imported *translation.ImportResult, candidate *planpb.Expr) error {
	if f.stage == StageCandidateValidation {
		if f.cancel != nil {
			f.cancel()
		}
		candidate.IsTemplate = !candidate.IsTemplate
	}
	return f.pairedTranslator.ValidateCandidate(ctx, imported, candidate)
}

func TestRoundTripStageFailures(t *testing.T) {
	input := gateInput(t)
	before := proto.Clone(input.Original)
	for _, stage := range []Stage{StageSetup, StageImport, StageExport, StageCandidateValidation} {
		for _, cancelled := range []bool{false, true} {
			ctx, cancel := context.WithCancel(context.Background())
			build := func(in AttemptInput, runtime *adapterruntime.Runtime) (pairedTranslator, error) {
				if stage == StageSetup {
					if cancelled {
						cancel()
					}
					return nil, merr.ErrServiceInternal
				}
				translator, err := newTranslator(in, runtime)
				if err != nil {
					return nil, err
				}
				fault := &faultTranslator{pairedTranslator: translator, stage: stage}
				if cancelled {
					fault.cancel = cancel
				}
				return fault, nil
			}
			result, err := roundTrip(ctx, input, func(*planpb.Expr) error { t.Fatal("failed stage reached placement"); return nil }, adapterruntime.SharedRuntime, build)
			require.Nil(t, result.Candidate())
			require.Equal(t, stage, result.Report().Stage)
			require.True(t, proto.Equal(before, input.Original))
			if cancelled || stage == StageSetup {
				require.True(t, IsAdapterAbort(err))
				require.Equal(t, OutcomeUnknown, result.Outcome())
				if cancelled {
					require.ErrorIs(t, err, context.Canceled)
				} else {
					require.ErrorIs(t, err, merr.ErrServiceInternal)
				}
			} else {
				require.NoError(t, err)
				require.Equal(t, OutcomeRetainedAfterFailure, result.Outcome())
			}
			cancel()
		}
	}
	// An ordinary internal code alone cannot establish containment of a late failure.
	cause := merr.WrapErrServiceInternalMsg("unclassified export failure")
	result, err := roundTrip(context.Background(), input, func(*planpb.Expr) error { t.Fatal("uncontained failure reached placement"); return nil }, adapterruntime.SharedRuntime, func(in AttemptInput, runtime *adapterruntime.Runtime) (pairedTranslator, error) {
		translator, err := newTranslator(in, runtime)
		return &faultTranslator{pairedTranslator: translator, stage: StageExport, unclassified: cause}, err
	})
	require.True(t, IsAdapterAbort(err))
	require.ErrorIs(t, err, cause)
	require.Nil(t, result.Candidate())
}

// Counts context entry independently of implementation dispatch.
type executionCounter struct {
	functioninterfaces.FunctionContext
	calls int
}

func (c *executionCounter) Evaluate(handle functioninterfaces.FunctionHandle, signature *shared.FunctionSignature, args []valueinterfaces.Value) (valueinterfaces.Value, error) {
	c.calls++
	return c.FunctionContext.Evaluate(handle, signature, args)
}

// Shares production metadata/mapping while selecting isolated execution bindings.
type instrumentedRuntime struct {
	translation.Runtime
	provider functioninterfaces.FunctionContext
}

func (r *instrumentedRuntime) FunctionContext() functioninterfaces.FunctionContext { return r.provider }

func TestRoundTripDoesNotExecute(t *testing.T) {
	input := gateInput(t)
	field := &schemapb.FieldSchema{FieldID: 101, Name: "i", DataType: schemapb.DataType_Int64}
	helper, err := typeutil.CreateSchemaHelper(&schemapb.CollectionSchema{Fields: []*schemapb.FieldSchema{field}})
	require.NoError(t, err)
	input.SchemaView, err = schema.NewView(helper, []*schemapb.FieldSchema{field})
	require.NoError(t, err)
	column := &planpb.ColumnInfo{FieldId: 101, DataType: schemapb.DataType_Int64}
	input.Original = &planpb.Expr{Expr: &planpb.Expr_BinaryExpr{BinaryExpr: &planpb.BinaryExpr{
		Op:    planpb.BinaryExpr_LogicalAnd,
		Left:  &planpb.Expr{Expr: &planpb.Expr_UnaryRangeExpr{UnaryRangeExpr: &planpb.UnaryRangeExpr{ColumnInfo: column, Op: planpb.OpType_Equal, Value: &planpb.GenericValue{Val: &planpb.GenericValue_Int64Val{Int64Val: 7}}}}},
		Right: &planpb.Expr{Expr: &planpb.Expr_NullExpr{NullExpr: &planpb.NullExpr{ColumnInfo: column, Op: planpb.NullExpr_IsNotNull}}},
	}}}
	runtime, err := adapterruntime.NewRuntime()
	require.NoError(t, err)
	p := runtime.Types()
	builder := functions.NewFunctionContextBuilder(runtime.Catalog())
	implementations := 0
	for _, typ := range []types.Type{p.BoolNonNull, p.Int8NonNull, p.Int16NonNull, p.Int32NonNull, p.Int64NonNull, p.Float32NonNull, p.Float64NonNull} {
		ops := []planpb.OpType{planpb.OpType_Equal, planpb.OpType_NotEqual}
		if typ != p.BoolNonNull {
			ops = append(ops, planpb.OpType_LessThan, planpb.OpType_LessEqual, planpb.OpType_GreaterThan, planpb.OpType_GreaterEqual)
		}
		for _, op := range ops {
			handle, err := runtime.ComparisonFunction(op, typ, typ)
			require.NoError(t, err)
			require.NoError(t, builder.BindImplementation(handle, func([]valueinterfaces.Value) (valueinterfaces.Value, error) {
				implementations++
				t.Fatal("round trip executed a function")
				return nil, nil
			}))
		}
	}
	provider, err := builder.Build()
	require.NoError(t, err)
	counter := &executionCounter{FunctionContext: provider}
	var importedCount, exportedCount int
	result, err := roundTrip(context.Background(), input, func(candidate *planpb.Expr) error {
		require.True(t, proto.Equal(input.Original, candidate))
		return nil
	}, func() (*adapterruntime.Runtime, error) { return runtime, nil }, func(in AttemptInput, runtime *adapterruntime.Runtime) (pairedTranslator, error) {
		translator, err := translation.New(&instrumentedRuntime{Runtime: runtime, provider: counter}, in.SchemaView, p)
		return &traversalCounter{pairedTranslator: translator, imports: &importedCount, exports: &exportedCount}, err
	})
	require.NoError(t, err)
	require.Equal(t, OutcomeTranslated, result.Outcome())
	require.Equal(t, 1, importedCount)
	require.Equal(t, 1, exportedCount)
	require.Zero(t, counter.calls)
	require.Zero(t, implementations)
}

// Records completed real traversals, without substituting a source copy for either stage.
type traversalCounter struct {
	pairedTranslator
	imports *int
	exports *int
}

func (c *traversalCounter) Import(ctx context.Context, admitted *translation.AdmissionPlan) (*translation.ImportResult, error) {
	result, err := c.pairedTranslator.Import(ctx, admitted)
	if err == nil {
		*c.imports++
	}
	return result, err
}

func (c *traversalCounter) Export(ctx context.Context, imported *translation.ImportResult, graph expressions.Expression) (*planpb.Expr, error) {
	result, err := c.pairedTranslator.Export(ctx, imported, graph)
	if err == nil {
		*c.exports++
	}
	return result, err
}
