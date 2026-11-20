package runtime

import (
	"github.com/lie-yan/predicate-optimization/foundation"
	"github.com/lie-yan/predicate-optimization/types"
)

// Shared primitive descriptors are immutable after initialization. Their
// availability does not expand the translator's admitted field domains.
var (
	boolNonNullType    = mustPrimitiveType(types.NewBoolType(foundation.NonNull))
	boolMayNullType    = mustPrimitiveType(types.NewBoolType(foundation.MayNull))
	int8NonNullType    = mustPrimitiveType(types.NewIntType(types.Int8, foundation.NonNull))
	int8MayNullType    = mustPrimitiveType(types.NewIntType(types.Int8, foundation.MayNull))
	int16NonNullType   = mustPrimitiveType(types.NewIntType(types.Int16, foundation.NonNull))
	int16MayNullType   = mustPrimitiveType(types.NewIntType(types.Int16, foundation.MayNull))
	int32NonNullType   = mustPrimitiveType(types.NewIntType(types.Int32, foundation.NonNull))
	int32MayNullType   = mustPrimitiveType(types.NewIntType(types.Int32, foundation.MayNull))
	int64NonNullType   = mustPrimitiveType(types.NewIntType(types.Int64, foundation.NonNull))
	int64MayNullType   = mustPrimitiveType(types.NewIntType(types.Int64, foundation.MayNull))
	float32NonNullType = mustPrimitiveType(types.NewFloatType(types.Float32, foundation.NonNull))
	float32MayNullType = mustPrimitiveType(types.NewFloatType(types.Float32, foundation.MayNull))
	float64NonNullType = mustPrimitiveType(types.NewFloatType(types.Float64, foundation.NonNull))
	float64MayNullType = mustPrimitiveType(types.NewFloatType(types.Float64, foundation.MayNull))
)

// Used only for fixed, valid primitive declarations above, never request data.
func mustPrimitiveType[T types.Type](typ T, err error) T {
	if err != nil {
		panic(err)
	}
	return typ
}

// Value snapshot of shared primitive descriptors; all referenced Types remain immutable.
type PrimitiveTypes struct {
	// Non-null Bool descriptor.
	BoolNonNull *types.BoolType
	// Nullable Bool descriptor.
	BoolMayNull *types.BoolType
	// Non-null Int8 descriptor.
	Int8NonNull *types.IntType
	// Nullable Int8 descriptor.
	Int8MayNull *types.IntType
	// Non-null Int16 descriptor.
	Int16NonNull *types.IntType
	// Nullable Int16 descriptor.
	Int16MayNull *types.IntType
	// Non-null Int32 descriptor.
	Int32NonNull *types.IntType
	// Nullable Int32 descriptor.
	Int32MayNull *types.IntType
	// Non-null Int64 descriptor.
	Int64NonNull *types.IntType
	// Nullable Int64 descriptor.
	Int64MayNull *types.IntType
	// Non-null Float32 descriptor.
	Float32NonNull *types.FloatType
	// Nullable Float32 descriptor.
	Float32MayNull *types.FloatType
	// Non-null Float64 descriptor.
	Float64NonNull *types.FloatType
	// Nullable Float64 descriptor.
	Float64MayNull *types.FloatType
}

// Collects the existing shared instances without allocating new Type descriptors.
func sharedPrimitiveTypes() PrimitiveTypes {
	return PrimitiveTypes{
		BoolNonNull:    boolNonNullType,
		BoolMayNull:    boolMayNullType,
		Int8NonNull:    int8NonNullType,
		Int8MayNull:    int8MayNullType,
		Int16NonNull:   int16NonNullType,
		Int16MayNull:   int16MayNullType,
		Int32NonNull:   int32NonNullType,
		Int32MayNull:   int32MayNullType,
		Int64NonNull:   int64NonNullType,
		Int64MayNull:   int64MayNullType,
		Float32NonNull: float32NonNullType,
		Float32MayNull: float32MayNullType,
		Float64NonNull: float64NonNullType,
		Float64MayNull: float64MayNullType,
	}
}
