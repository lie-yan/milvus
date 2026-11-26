package functions

import (
	"github.com/lie-yan/predicate-optimization/types"
)

// Registration-ordered candidates, with a fast path for scalar calls of up to two arguments.
type overloadSet struct {
	// Canonical entry positions for general binding.
	entries []int
	// Argument signatures ignoring outer nullability, never result Types.
	scalar map[scalarSignature][]int
	// Every registered signature can be represented by the scalar index.
	scalarOnly bool
}

// Comparable structural signature for the common zero-, unary- and binary-call paths.
type scalarSignature struct {
	// Separates zero arguments from absent trailing slots.
	arity int
	// Ordered argument domains; outer nullability is handled by the binder.
	args [2]scalarArgument
}

// Concrete scalar identity without pointer identity or display-string encoding.
type scalarArgument struct {
	// Bool, integer, floating-point or VARCHAR variant.
	kind byte
	// Integer/float width, or VARCHAR byte bound.
	size uint64
	// Distinguishes unbounded VARCHAR from a concrete bound.
	bounded bool
}

// Returns false for domains requiring the general overload binder.
func scalarLookupKey(args []types.Type) (scalarSignature, bool) {
	key := scalarSignature{arity: len(args)}
	if len(args) > len(key.args) {
		return key, false
	}
	for i, arg := range args {
		switch t := arg.(type) {
		case *types.BoolType:
			key.args[i] = scalarArgument{kind: 'b'}
		case *types.IntType:
			key.args[i] = scalarArgument{kind: 'i', size: uint64(t.IntegerWidth())}
		case *types.FloatType:
			key.args[i] = scalarArgument{kind: 'f', size: uint64(t.FloatWidth())}
		case *types.VarCharType:
			size, bounded := t.MaxBytes()
			key.args[i] = scalarArgument{kind: 's', size: size, bounded: bounded}
		default:
			return key, false
		}
	}
	return key, true
}

// Builds immutable candidate lists without changing canonical handle positions.
func addOverload[K comparable](index map[K]*overloadSet, key K, position int, args []types.Type) {
	set := index[key]
	if set == nil {
		set = &overloadSet{scalar: make(map[scalarSignature][]int), scalarOnly: true}
		index[key] = set
	}
	set.entries = append(set.entries, position)
	signature, ok := scalarLookupKey(args)
	if !ok {
		set.scalarOnly = false
		return
	}
	set.scalar[signature] = append(set.scalar[signature], position)
}

// Preserves general binding whenever either side contains an unindexed Type shape.
func (s *overloadSet) candidates(args []types.Type) []int {
	if s == nil {
		return nil
	}
	if s.scalarOnly {
		if key, ok := scalarLookupKey(args); ok {
			return s.scalar[key]
		}
	}
	return s.entries
}
