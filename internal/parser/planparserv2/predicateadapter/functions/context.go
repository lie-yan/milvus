package functions

import (
	"github.com/lie-yan/predicate-optimization/foundation"
	shared "github.com/lie-yan/predicate-optimization/functions"
	functioninterfaces "github.com/lie-yan/predicate-optimization/functions/interfaces"
	"github.com/lie-yan/predicate-optimization/types"
	valueinterfaces "github.com/lie-yan/predicate-optimization/values/interfaces"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/internal/assertions"
)

// Immutable provider sharing catalog identities and owning execution bindings; concurrent evaluation additionally requires thread-safe implementations.
type FunctionContext struct {
	// Frozen catalog and implementation associations, never replaced after construction.
	bindings *implementationBindings
}

var _ functioninterfaces.FunctionContext = (*FunctionContext)(nil)

// Sequential context setup whose successful Build consumes every copy.
type FunctionContextBuilder struct {
	// Shared private lifecycle and canonical implementation bindings.
	bindings implementationBindingsBuilder
}

// Starts execution setup around one completed catalog without rebuilding its definitions.
func NewFunctionContextBuilder(catalog *FunctionCatalog) *FunctionContextBuilder {
	return &FunctionContextBuilder{bindings: *newImplementationBindingsBuilder(catalog)}
}

// Binds a non-nil implementation once per canonical entry; omitted entries remain unavailable for dispatch.
func (b *FunctionContextBuilder) BindImplementation(handle functioninterfaces.FunctionHandle, implementation FunctionImplementation) error {
	assertions.NotNilPointer(b)
	return b.bindings.bind(handle, implementation)
}

// Freezes execution setup and returns a complete provider retaining the original catalog.
func (b *FunctionContextBuilder) Build() (*FunctionContext, error) {
	assertions.NotNilPointer(b)
	bindings, err := b.bindings.build()
	if err != nil {
		return nil, err
	}
	return &FunctionContext{bindings: bindings}, nil
}

// Resolves a built-in through the shared catalog independently of execution availability.
func (c *FunctionContext) LookupBuiltInFunction(name shared.BuiltInFunctionName, args []types.Type) (functioninterfaces.FunctionHandle, error) {
	assertions.NotNilPointer(c)
	return c.bindings.catalog.LookupBuiltInFunction(name, args)
}

// Resolves an arithmetic operator through the catalog's family, arity and binding checks.
func (c *FunctionContext) ArithmeticFunction(operator foundation.OperatorType, args []types.Type) (functioninterfaces.FunctionHandle, error) {
	assertions.NotNilPointer(c)
	return c.bindings.catalog.ArithmeticFunction(operator, args)
}

// Resolves a binary comparison using the catalog's exact domain and outer-nullability rules.
func (c *FunctionContext) ComparisonFunction(operator foundation.OperatorType, left, right types.Type) (functioninterfaces.FunctionHandle, error) {
	assertions.NotNilPointer(c)
	return c.bindings.catalog.ComparisonFunction(operator, left, right)
}

// Classifies a validated catalog handle without invoking its implementation.
func (c *FunctionContext) IsArithmeticFunction(handle functioninterfaces.FunctionHandle) (bool, error) {
	assertions.NotNilPointer(c)
	return c.bindings.catalog.IsArithmeticFunction(handle)
}

// Recognizes comparison metadata while preserving invalid-handle errors.
func (c *FunctionContext) IsComparisonFunction(handle functioninterfaces.FunctionHandle) (bool, error) {
	assertions.NotNilPointer(c)
	return c.bindings.catalog.IsComparisonFunction(handle)
}

// Returns the original immutable metadata after catalog ownership validation.
func (c *FunctionContext) Metadata(handle functioninterfaces.FunctionHandle) (*shared.FunctionMetadata, error) {
	assertions.NotNilPointer(c)
	return c.bindings.catalog.Metadata(handle)
}

// Executes through this context's frozen bindings with signature, Value and null-policy checks.
func (c *FunctionContext) Evaluate(handle functioninterfaces.FunctionHandle, signature *shared.FunctionSignature, args []valueinterfaces.Value) (valueinterfaces.Value, error) {
	assertions.NotNilPointer(c)
	return c.bindings.evaluate(handle, signature, args)
}
