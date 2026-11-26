package runtime

import (
	"github.com/lie-yan/predicate-optimization/foundation"
	shared "github.com/lie-yan/predicate-optimization/functions"
	functioninterfaces "github.com/lie-yan/predicate-optimization/functions/interfaces"
	"github.com/lie-yan/predicate-optimization/types"
	valueinterfaces "github.com/lie-yan/predicate-optimization/values/interfaces"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/functions"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/internal/assertions"
	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/values"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Complete immutable foundation; construction does not publish it or enable production callers.
type Runtime struct {
	// Selected stateless factory, captured once during construction.
	factory valueinterfaces.ValueFactory
	// Shared immutable primitive descriptors returned as a value snapshot.
	types PrimitiveTypes
	// Canonical definitions owned by this runtime instance.
	catalog *functions.FunctionCatalog
	// Matching immutable provider with placeholder implementations.
	context *functions.FunctionContext
}

// Constructs an isolated empty-pipeline bundle; later publication must reuse it across sessions.
func NewRuntime() (*Runtime, error) {
	definitions, err := functions.ComparisonDefinitions(int64NonNullType, boolNonNullType)
	if err != nil {
		return nil, err
	}
	numeric, err := functions.NumericComparisonDefinitions(boolNonNullType, int8NonNullType, int16NonNullType, int32NonNullType, float32NonNullType, float64NonNullType)
	if err != nil {
		return nil, err
	}
	return assembleRuntime(append(definitions, numeric...))
}

// Registers declarations and binds placeholders before any complete bundle can escape.
func assembleRuntime(definitions []functions.FunctionDefinition) (*Runtime, error) {
	builder := functions.NewFunctionCatalogBuilder()
	for _, definition := range definitions {
		if err := builder.Register(definition); err != nil {
			return nil, err
		}
	}
	catalog, err := builder.Build()
	if err != nil {
		return nil, err
	}
	contextBuilder := functions.NewFunctionContextBuilder(catalog)
	for _, definition := range definitions {
		handle, err := catalog.LookupFunction(definition.Name, definition.Metadata.Signature().ArgumentTypes())
		if err != nil {
			return nil, merr.Wrap(err, "predicateadapter: runtime definition lookup")
		}
		metadata, err := catalog.Metadata(handle)
		if err != nil {
			return nil, err
		}
		op, present := metadata.Operator().Get()
		if _, supported := milvusComparison(op); !present || !supported {
			return nil, merr.WrapErrServiceInternalMsg("predicateadapter: runtime definition has no comparison mapping")
		}
		signature := metadata.Signature()
		if signature.ArgumentCount() != 2 {
			return nil, merr.WrapErrServiceInternalMsg("predicateadapter: comparison definition is not binary")
		}
		resolved, err := catalog.ComparisonFunction(op, signature.ArgumentTypeAt(0), signature.ArgumentTypeAt(1))
		if err != nil {
			return nil, merr.Wrap(err, "predicateadapter: runtime comparison resolution")
		}
		if !handle.EqualsTo(resolved) {
			return nil, merr.WrapErrServiceInternalMsg("predicateadapter: comparison mapping resolves a different definition")
		}
		if err := contextBuilder.BindImplementation(handle, emptyPipelineImplementation); err != nil {
			return nil, err
		}
	}
	context, err := contextBuilder.Build()
	if err != nil {
		return nil, err
	}
	runtime := &Runtime{
		factory: values.Factory, types: sharedPrimitiveTypes(), catalog: catalog, context: context,
	}
	if err := runtime.validateComparisons(); err != nil {
		return nil, err
	}
	return runtime, nil
}

// Checks the complete translation domain before the immutable runtime can escape.
func (r *Runtime) validateComparisons() error {
	p := r.types
	for _, typ := range []types.Type{p.BoolNonNull, p.Int8NonNull, p.Int16NonNull, p.Int32NonNull, p.Int64NonNull, p.Float32NonNull, p.Float64NonNull} {
		ops := []planpb.OpType{planpb.OpType_Equal, planpb.OpType_NotEqual}
		if typ != p.BoolNonNull {
			ops = append(ops, planpb.OpType_LessThan, planpb.OpType_LessEqual, planpb.OpType_GreaterThan, planpb.OpType_GreaterEqual)
		}
		for _, op := range ops {
			handle, err := r.ComparisonFunction(op, typ, typ)
			if err != nil {
				return merr.Wrap(err, "predicateadapter: runtime comparison coverage")
			}
			metadata, err := r.context.Metadata(handle)
			if err != nil {
				return err
			}
			recovered, err := r.ComparisonOperator(handle)
			if err != nil {
				return err
			}
			signature := metadata.Signature()
			if recovered != op || signature.ArgumentCount() != 2 || !typ.EqualsTo(signature.ArgumentTypeAt(0)) || !typ.EqualsTo(signature.ArgumentTypeAt(1)) || !p.BoolNonNull.EqualsTo(signature.ResultType()) {
				return merr.WrapErrServiceInternalMsg("predicateadapter: inconsistent runtime comparison signature")
			}
		}
	}
	return nil
}

// Placeholder with unspecified invocation behavior; the empty pipeline must never execute it.
func emptyPipelineImplementation([]valueinterfaces.Value) (valueinterfaces.Value, error) {
	panic("predicateadapter: unexpected empty-pipeline implementation invocation")
}

// Returns the captured stateless factory.
func (r *Runtime) ValueFactory() valueinterfaces.ValueFactory {
	assertions.NotNilPointer(r)
	return r.factory
}

// Copies descriptor associations while sharing the immutable Types.
func (r *Runtime) Types() PrimitiveTypes { assertions.NotNilPointer(r); return r.types }

// Returns the immutable catalog for named and operator lookup.
func (r *Runtime) Catalog() *functions.FunctionCatalog { assertions.NotNilPointer(r); return r.catalog }

// Returns the exact provider to retain on calls; production empty translation must not evaluate.
func (r *Runtime) FunctionContext() functioninterfaces.FunctionContext {
	assertions.NotNilPointer(r)
	return r.context
}

// Resolves a Milvus comparison enum through the matching catalog without translating protobuf nodes.
func (r *Runtime) ComparisonFunction(op planpb.OpType, left, right types.Type) (functioninterfaces.FunctionHandle, error) {
	assertions.NotNilPointer(r)
	var operator foundation.OperatorType
	switch op {
	case planpb.OpType_Equal:
		operator = foundation.Equal
	case planpb.OpType_NotEqual:
		operator = foundation.NotEqual
	case planpb.OpType_LessThan:
		operator = foundation.LessThan
	case planpb.OpType_LessEqual:
		operator = foundation.LessThanOrEqual
	case planpb.OpType_GreaterThan:
		operator = foundation.GreaterThan
	case planpb.OpType_GreaterEqual:
		operator = foundation.GreaterThanOrEqual
	default:
		return nil, merr.Wrap(shared.ErrUnsupportedLookup, "predicateadapter: unsupported Milvus comparison")
	}
	return r.catalog.ComparisonFunction(operator, left, right)
}

// Recovers a Milvus comparison enum only after validating handle ownership and metadata.
func (r *Runtime) ComparisonOperator(handle functioninterfaces.FunctionHandle) (planpb.OpType, error) {
	assertions.NotNilPointer(r)
	metadata, err := r.catalog.Metadata(handle)
	if err != nil {
		return planpb.OpType_Invalid, err
	}
	op, present := metadata.Operator().Get()
	mapped, supported := milvusComparison(op)
	if !present || !supported {
		return planpb.OpType_Invalid, merr.Wrap(shared.ErrUnsupportedLookup, "predicateadapter: function has no Milvus comparison mapping")
	}
	return mapped, nil
}

// Maps semantic comparison metadata without function-name or handle-position bookkeeping.
func milvusComparison(op foundation.OperatorType) (planpb.OpType, bool) {
	switch op {
	case foundation.Equal:
		return planpb.OpType_Equal, true
	case foundation.NotEqual:
		return planpb.OpType_NotEqual, true
	case foundation.LessThan:
		return planpb.OpType_LessThan, true
	case foundation.LessThanOrEqual:
		return planpb.OpType_LessEqual, true
	case foundation.GreaterThan:
		return planpb.OpType_GreaterThan, true
	case foundation.GreaterThanOrEqual:
		return planpb.OpType_GreaterEqual, true
	default:
		return planpb.OpType_Invalid, false
	}
}
