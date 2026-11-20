package translation

import (
	"context"

	"github.com/lie-yan/predicate-optimization/expressions"
	expressionutils "github.com/lie-yan/predicate-optimization/expressions/utils"
	"github.com/lie-yan/predicate-optimization/foundation"
	functioninterfaces "github.com/lie-yan/predicate-optimization/functions/interfaces"
	"github.com/lie-yan/predicate-optimization/types"
	valueinterfaces "github.com/lie-yan/predicate-optimization/values/interfaces"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/internal/assertions"
	adapterruntime "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/runtime"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/schema"
	translationutils "github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/translation/utils"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Immutable translation inputs; borrowed source/schema data must remain stable during an attempt.
type Translator struct {
	// Selected immutable runtime and its operation mapping.
	runtime Runtime
	// Captured factory for native Value construction.
	factory valueinterfaces.ValueFactory
	// Exact function provider for future native calls.
	context functioninterfaces.FunctionContext
	// Pinned schema membership and metadata.
	view *schema.View
	// Value snapshot of shared immutable primitive descriptors.
	primitives adapterruntime.PrimitiveTypes
}

// Successful whole-source routing with no optimizer graph or registered identities.
type AdmissionPlan struct {
	// Exact translator that established the routes.
	owner *Translator
	// Immutable selected source positions.
	routes []admittedRoute
	// Complete predicate's route position.
	root routeID
}

// One import's immutable graph associations and private preservation registry.
type ImportResult struct {
	// Successful routing and pinned translator inputs.
	admitted *AdmissionPlan
	// Attempt-local preservation identities.
	opaques *translationutils.OpaqueRegistry
	// Lazily allocated on the first admitted field-bearing import.
	fields *schema.FieldRegistry
	// Source/graph positions, including distinct repeated occurrences.
	occurrences []occurrence
	// Complete graph's source position.
	root occurrenceID
}

// Captures capabilities and verifies that the shared Types and comparison mappings agree.
func New(runtime Runtime, view *schema.View, primitives adapterruntime.PrimitiveTypes) (*Translator, error) {
	integer, boolean := primitives.Int64NonNull, primitives.BoolNonNull
	if runtime == nil || view == nil || integer == nil || boolean == nil {
		return nil, merr.WrapErrServiceInternalMsg("predicate translation: missing setup input")
	}
	assertions.NotNil(runtime)
	if integer.IntegerWidth() != types.Int64 || integer.Nullability() != foundation.NonNull || boolean.Nullability() != foundation.NonNull {
		return nil, merr.WrapErrServiceInternalMsg("predicate translation: expected NonNull Int64/Bool descriptors")
	}
	// Every admitted numeric domain must retain its declared width and non-null Type.
	for _, slot := range []struct {
		typ   *types.IntType
		width types.IntWidth
	}{
		{primitives.Int8NonNull, types.Int8}, {primitives.Int16NonNull, types.Int16},
		{primitives.Int32NonNull, types.Int32}, {integer, types.Int64},
	} {
		if slot.typ == nil || slot.typ.IntegerWidth() != slot.width || slot.typ.Nullability() != foundation.NonNull {
			return nil, merr.WrapErrServiceInternalMsg("predicate translation: inconsistent integer descriptor")
		}
	}
	for _, slot := range []struct {
		typ   *types.FloatType
		width types.FloatWidth
	}{
		{primitives.Float32NonNull, types.Float32}, {primitives.Float64NonNull, types.Float64},
	} {
		if slot.typ == nil || slot.typ.FloatWidth() != slot.width || slot.typ.Nullability() != foundation.NonNull {
			return nil, merr.WrapErrServiceInternalMsg("predicate translation: inconsistent floating descriptor")
		}
	}
	t := &Translator{runtime: runtime, factory: runtime.ValueFactory(), context: runtime.FunctionContext(), view: view, primitives: primitives}
	if t.factory == nil || t.context == nil {
		return nil, merr.WrapErrServiceInternalMsg("predicate translation: missing runtime capabilities")
	}
	assertions.NotNil(t.factory)
	assertions.NotNil(t.context)
	for _, typ := range []types.Type{boolean, primitives.Int8NonNull, primitives.Int16NonNull, primitives.Int32NonNull, integer, primitives.Float32NonNull, primitives.Float64NonNull} {
		ops := []planpb.OpType{planpb.OpType_Equal, planpb.OpType_NotEqual}
		if typ != boolean {
			ops = append(ops, planpb.OpType_LessThan, planpb.OpType_LessEqual, planpb.OpType_GreaterThan, planpb.OpType_GreaterEqual)
		}
		for _, op := range ops {
			handle, err := runtime.ComparisonFunction(op, typ, typ)
			if err != nil {
				return nil, merr.Wrap(err, "predicate translation: setup comparison lookup")
			}
			metadata, err := t.context.Metadata(handle)
			if err != nil {
				return nil, merr.Wrap(err, "predicate translation: setup context ownership")
			}
			recovered, err := runtime.ComparisonOperator(handle)
			if err != nil {
				return nil, merr.Wrap(err, "predicate translation: setup reverse mapping")
			}
			if metadata == nil || recovered != op {
				return nil, merr.WrapErrServiceInternalMsg("predicate translation: inconsistent comparison mapping")
			}
			signature := metadata.Signature()
			if signature == nil || signature.ArgumentCount() != 2 || !typ.EqualsTo(signature.ArgumentTypeAt(0)) || !typ.EqualsTo(signature.ArgumentTypeAt(1)) || !boolean.EqualsTo(signature.ResultType()) {
				return nil, merr.WrapErrServiceInternalMsg("predicate translation: inconsistent comparison signature")
			}
		}
	}
	return t, nil
}

// Selects complete native/opaque routing before creating any graph or registry.
func (t *Translator) Admit(ctx context.Context, source *planpb.Expr) (*AdmissionPlan, error) {
	a := &admitter{ctx: ctx, translator: t}
	result, err := a.admitExpr(source)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if result.limitation != noLimitation {
		return nil, newFailure(UnsupportedSource, StageAdmission, merr.Wrapf(merr.ErrServiceUnimplemented, "predicate translation: admission limitation %d", result.limitation))
	}
	return &AdmissionPlan{owner: t, routes: a.routes, root: result.route}, nil
}

// Consumes approved routes without reconsidering fallback and returns only a complete graph.
func (t *Translator) Import(ctx context.Context, admitted *AdmissionPlan) (*ImportResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if admitted == nil || admitted.owner != t || admitted.root < 0 || int(admitted.root) >= len(admitted.routes) {
		return nil, translationDefect(StageImport, "foreign or incomplete admission")
	}
	registry, err := translationutils.NewOpaqueRegistry()
	if err != nil {
		return nil, newFailure(TranslationDefect, StageImport, err)
	}
	result := &ImportResult{admitted: admitted, opaques: registry}
	visitor := &importer{ctx: ctx, result: result}
	root, err := visitor.importExpr(admitted.root)
	if err != nil {
		return nil, err
	}
	result.root = root
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return result, nil
}

// Returns the immutable imported root; mutable association tables remain private.
func (r *ImportResult) Root() expressions.Expression {
	assertions.NotNilPointer(r)
	return r.occurrences[r.root].imported
}

// Reconstructs a complete private protobuf from the supplied graph and owned associations.
func (t *Translator) Export(ctx context.Context, imported *ImportResult, graph expressions.Expression) (*planpb.Expr, error) {
	if err := t.checkGraph(ctx, imported, graph, StageExport); err != nil {
		return nil, err
	}
	visitor := &exporter{ctx: ctx, imported: imported}
	candidate, err := visitor.exportExpr(imported.root, graph)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return candidate, nil
}

// Inspects actual emitted data independently of exporter success, without accepting or attaching it.
func (t *Translator) ValidateCandidate(ctx context.Context, imported *ImportResult, candidate *planpb.Expr) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := t.checkOwner(imported, StageCandidateValidation); err != nil {
		return err
	}
	visitor := &candidateChecker{ctx: ctx, imported: imported}
	if err := visitor.checkExpr(imported.root, candidate); err != nil {
		return err
	}
	return ctx.Err()
}

// Rejects missing or foreign state before accessing attempt-local tables.
func (t *Translator) checkOwner(imported *ImportResult, stage Stage) error {
	if imported == nil || imported.admitted == nil || imported.admitted.owner != t || imported.opaques == nil || imported.root < 0 || int(imported.root) >= len(imported.occurrences) {
		return translationDefect(stage, "foreign or incomplete imported state")
	}
	return nil
}

// Establishes framework validity before form-specific traversal; the framework helper itself is synchronous.
func (t *Translator) checkGraph(ctx context.Context, imported *ImportResult, graph expressions.Expression, stage Stage) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := t.checkOwner(imported, stage); err != nil {
		return err
	}
	if err := expressionutils.ValidateForOptimization(graph); err != nil {
		return newFailure(TranslationDefect, stage, err)
	}
	visitor := &graphChecker{ctx: ctx, imported: imported, stage: stage}
	return visitor.checkExpr(imported.root, graph)
}

// Marks an internal invariant failure at its consuming stage without a new wire error code.
func translationDefect(stage Stage, message string) error {
	return newFailure(TranslationDefect, stage, merr.WrapErrServiceInternalMsg("predicate translation: %s", message))
}
