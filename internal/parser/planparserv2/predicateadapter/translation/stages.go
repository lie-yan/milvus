package translation

import (
	"context"

	"github.com/lie-yan/predicate-optimization/expressions"
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
	// Shared descriptors used by the currently admitted native domains.
	primitives primitiveTypes
}

// Descriptor associations needed during translation; the runtime retains the full set.
type primitiveTypes struct {
	BoolMayNull                                           *types.BoolType
	Int64MayNull                                          *types.IntType
	BoolNonNull                                           *types.BoolType
	Int8NonNull, Int16NonNull, Int32NonNull, Int64NonNull *types.IntType
	Float32NonNull, Float64NonNull                        *types.FloatType
}

// One imported graph and its field/opaque bindings.
type ImportResult struct {
	// Translator owning this graph and its pinned inputs.
	owner *Translator
	// Allocated on the first admitted opaque region.
	opaques *translationutils.OpaqueRegistry
	// Lazily allocated on the first admitted field-bearing import.
	fields *schema.FieldRegistry
	// Complete immutable expression graph.
	root expressions.Expression
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
	if primitives.BoolMayNull == nil || primitives.BoolMayNull.Nullability() != foundation.MayNull || primitives.Int64MayNull == nil || primitives.Int64MayNull.IntegerWidth() != types.Int64 || primitives.Int64MayNull.Nullability() != foundation.MayNull {
		return nil, merr.WrapErrServiceInternalMsg("predicate translation: expected nullable Int64/Bool descriptors")
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
	t := &Translator{runtime: runtime, factory: runtime.ValueFactory(), context: runtime.FunctionContext(), view: view, primitives: primitiveTypes{
		BoolNonNull: boolean, BoolMayNull: primitives.BoolMayNull, Int64MayNull: primitives.Int64MayNull,
		Int8NonNull: primitives.Int8NonNull, Int16NonNull: primitives.Int16NonNull,
		Int32NonNull: primitives.Int32NonNull, Int64NonNull: integer,
		Float32NonNull: primitives.Float32NonNull, Float64NonNull: primitives.Float64NonNull,
	}}
	if t.factory == nil || t.context == nil {
		return nil, merr.WrapErrServiceInternalMsg("predicate translation: missing runtime capabilities")
	}
	assertions.NotNil(t.factory)
	assertions.NotNil(t.context)
	// Only the concrete immutable runtime has already checked the complete catalog.
	// Wrappers may override capabilities and must still satisfy the setup contract.
	if _, validated := runtime.(*adapterruntime.Runtime); validated {
		return t, nil
	}
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

// Translates the source in one traversal; a failed import returns no partial graph or associations.
func (t *Translator) Import(ctx context.Context, source *planpb.Expr) (*ImportResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	result := &ImportResult{owner: t}
	visitor := &importer{ctx: ctx, result: result}
	root, err := visitor.importExpr(source)
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
	return r.root
}

// Reconstructs a private protobuf from a valid graph paired with its import associations.
// Graph validity and correct translation are caller preconditions; no graph-validation pass runs here.
func (t *Translator) Export(ctx context.Context, imported *ImportResult, graph expressions.Expression) (*planpb.Expr, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := t.checkOwner(imported, StageExport); err != nil {
		return nil, err
	}
	visitor := &exporter{ctx: ctx, imported: imported}
	candidate, err := visitor.exportExpr(graph)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return candidate, nil
}

// Rejects missing or foreign state before accessing attempt-local tables.
func (t *Translator) checkOwner(imported *ImportResult, stage Stage) error {
	if imported == nil || imported.owner != t || imported.root == nil {
		return translationDefect(stage, "foreign or incomplete imported state")
	}
	return nil
}

// Marks an internal invariant failure at its consuming stage without a new wire error code.
func translationDefect(stage Stage, message string) error {
	return newFailure(TranslationDefect, stage, merr.WrapErrServiceInternalMsg("predicate translation: %s", message))
}
