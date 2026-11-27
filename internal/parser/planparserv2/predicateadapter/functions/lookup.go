package functions

import (
	"errors"

	"github.com/lie-yan/predicate-optimization/foundation"
	shared "github.com/lie-yan/predicate-optimization/functions"
	functioninterfaces "github.com/lie-yan/predicate-optimization/functions/interfaces"
	"github.com/lie-yan/predicate-optimization/types"

	"github.com/milvus-io/milvus/internal/parser/planparserv2/predicateadapter/internal/assertions"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Resolves an exact qualified name using argument Types only, without evaluation or coercion.
func (c *FunctionCatalog) LookupFunction(name FunctionName, args []types.Type) (functioninterfaces.FunctionHandle, error) {
	c.assertLookupInputs(args)
	return c.lookup(args, c.byName[name])
}

// Resolves a registered built-in identity independently of semantic and diagnostic names.
func (c *FunctionCatalog) LookupBuiltInFunction(name shared.BuiltInFunctionName, args []types.Type) (functioninterfaces.FunctionHandle, error) {
	c.assertLookupInputs(args)
	return c.lookup(args, c.byBuiltIn[name])
}

// Resolves a binary comparison after validating the requested operator family.
func (c *FunctionCatalog) ComparisonFunction(op foundation.OperatorType, left, right types.Type) (functioninterfaces.FunctionHandle, error) {
	args := []types.Type{left, right}
	c.assertLookupInputs(args)
	if !op.IsComparisonOperator() {
		return nil, merr.Wrap(shared.ErrUnsupportedLookup, "predicateadapter: invalid comparison operator")
	}
	return c.lookupOperator(op, args)
}

// Resolves unary Negate or binary arithmetic after validating family and arity.
func (c *FunctionCatalog) ArithmeticFunction(op foundation.OperatorType, args []types.Type) (functioninterfaces.FunctionHandle, error) {
	c.assertLookupInputs(args)
	arity := 2
	if op == foundation.Negate {
		arity = 1
	}
	if !op.IsArithmeticOperator() || len(args) != arity {
		return nil, merr.Wrap(shared.ErrUnsupportedLookup, "predicateadapter: invalid arithmetic operator or arity")
	}
	return c.lookupOperator(op, args)
}

// Filters operator metadata through the same binder used by named and built-in lookup.
func (c *FunctionCatalog) lookupOperator(op foundation.OperatorType, args []types.Type) (functioninterfaces.FunctionHandle, error) {
	return c.lookup(args, c.byOperator[op])
}

// Checks programmer preconditions before route validation, even when no candidate exists.
func (c *FunctionCatalog) assertLookupInputs(args []types.Type) {
	assertions.NotNilPointer(c)
	for _, arg := range args {
		assertions.NotNil(arg)
	}
}

// Selects a unique binding among matching entries for every lookup route.
func (c *FunctionCatalog) lookup(args []types.Type, overloads *overloadSet) (functioninterfaces.FunctionHandle, error) {
	var selected *functionHandle
	for _, index := range overloads.candidates(args) {
		entry := c.entries[index]
		check := shared.ProbeFunctionBinding(entry.metadata, args)
		if !check.Supported() {
			if check.Is(shared.ErrContractViolation) {
				return nil, merr.Wrap(check.Err(), "predicateadapter: invalid catalog binding")
			}
			if check.Is(shared.ErrUnsupportedLookup) {
				continue
			}
			return nil, merr.Wrap(errors.Join(shared.ErrContractViolation, check.Err()), "predicateadapter: invalid catalog binding")
		}
		if selected != nil {
			return nil, merr.Wrap(shared.ErrUnsupportedLookup, "predicateadapter: ambiguous function binding")
		}
		selected = c.handleAt(index)
	}
	if overloads != nil {
		for _, index := range overloads.parameterized {
			entry := c.entries[index]
			if entry.declaration.Signature.ArgumentCount() != len(args) {
				continue
			}
			metadata, err := entry.declaration.bind(args)
			if err != nil {
				if errors.Is(err, shared.ErrUnsupportedLookup) {
					continue
				}
				return nil, err
			}
			if err := shared.CheckFunctionBinding(metadata, args); err != nil {
				if errors.Is(err, shared.ErrUnsupportedLookup) {
					continue
				}
				return nil, err
			}
			if selected != nil {
				return nil, merr.Wrap(shared.ErrUnsupportedLookup, "predicateadapter: ambiguous function binding")
			}
			selected = &functionHandle{catalog: c, index: index, metadata: metadata}
		}
	}
	if selected == nil {
		return nil, merr.Wrap(shared.ErrUnsupportedLookup, "predicateadapter: no matching function binding")
	}
	return selected, nil
}
