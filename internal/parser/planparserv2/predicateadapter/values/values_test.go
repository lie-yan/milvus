package values

import (
	"math"
	"testing"

	"github.com/lie-yan/predicate-optimization/foundation"
	"github.com/lie-yan/predicate-optimization/types"
	valueinterfaces "github.com/lie-yan/predicate-optimization/values/interfaces"
)

func checked[T any](t *testing.T, value T, err error) T {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	return value
}

func TestScalarFactoryExactPayloads(t *testing.T) {
	boolNonNullType, err := types.NewBoolType(foundation.NonNull)
	boolNonNullType = checked(t, boolNonNullType, err)
	int64NonNullType, err := types.NewIntType(types.Int64, foundation.NonNull)
	int64NonNullType = checked(t, int64NonNullType, err)

	for _, want := range []int64{math.MinInt64, -1, 0, 1, 9007199254740993, math.MaxInt64} {
		v, err := Factory.NewInteger(int64NonNullType, want)
		v = checked(t, v, err)
		payload, ok := v.(valueinterfaces.IntegerValue)
		if !ok || v.IsNull() || !v.Type().EqualsTo(int64NonNullType) || payload.Integer() != want {
			t.Fatalf("integer payload or Type changed: %d", want)
		}
	}
	for _, want := range []bool{false, true} {
		v, err := Factory.NewBool(boolNonNullType, want)
		v = checked(t, v, err)
		payload, ok := v.(valueinterfaces.BoolValue)
		if !ok || v.IsNull() || !v.Type().EqualsTo(boolNonNullType) || payload.Bool() != want {
			t.Fatalf("boolean payload or Type changed: %v", want)
		}
	}
}

// Independent immutable Value representation using public Type and payload interfaces for equality.
type scalarPeer struct {
	typ     types.Type
	integer int64
	boolean bool
}

func (v scalarPeer) Type() types.Type { return v.typ }
func (v scalarPeer) IsNull() bool     { return false }
func (v scalarPeer) Integer() int64   { return v.integer }
func (v scalarPeer) Bool() bool       { return v.boolean }
func (v scalarPeer) EqualsTo(other valueinterfaces.Value) bool {
	if other.IsNull() || !v.typ.EqualsTo(other.Type()) {
		return false
	}
	switch v.typ.(type) {
	case *types.BoolType:
		return v.boolean == other.(valueinterfaces.BoolValue).Bool()
	default:
		return v.integer == other.(valueinterfaces.IntegerValue).Integer()
	}
}
func (v scalarPeer) WithNullability(n foundation.Nullability) (valueinterfaces.Value, error) {
	typ, err := v.typ.WithNullability(n)
	if err != nil {
		return nil, err
	}
	v.typ = typ
	return v, nil
}

func TestScalarFactoryPublicEquality(t *testing.T) {
	boolNonNullType, err := types.NewBoolType(foundation.NonNull)
	boolNonNullType = checked(t, boolNonNullType, err)
	int64NonNullType, err := types.NewIntType(types.Int64, foundation.NonNull)
	int64NonNullType = checked(t, int64NonNullType, err)

	for _, peer := range []scalarPeer{
		{typ: boolNonNullType, boolean: false},
		{typ: int64NonNullType, integer: 9007199254740993},
	} {
		var v valueinterfaces.Value
		var err error
		if _, ok := peer.typ.(*types.BoolType); ok {
			v, err = Factory.NewBool(boolNonNullType, peer.boolean)
		} else {
			v, err = Factory.NewInteger(int64NonNullType, peer.integer)
		}
		v = checked(t, v, err)
		if !v.EqualsTo(peer) || !peer.EqualsTo(v) {
			t.Fatal("cross-representation equality is not symmetric")
		}
		changed := peer
		if _, ok := peer.typ.(*types.BoolType); ok {
			changed.boolean = !peer.boolean
		} else {
			changed.integer++
		}
		if v.EqualsTo(changed) || changed.EqualsTo(v) {
			t.Fatal("different payloads compare equal")
		}
		nullable, err := peer.WithNullability(foundation.MayNull)
		nullable = checked(t, nullable, err)
		if v.EqualsTo(nullable) || nullable.EqualsTo(v) {
			t.Fatal("equality discarded nullability")
		}
		adapted, err := v.WithNullability(foundation.MayNull)
		adapted = checked(t, adapted, err)
		if adapted.IsNull() || !adapted.EqualsTo(nullable) || !nullable.EqualsTo(adapted) {
			t.Fatal("adaptation changed payload")
		}
		if v.Type().Nullability() != foundation.NonNull {
			t.Fatal("adaptation mutated original")
		}
	}
	narrow, err := types.NewIntType(types.Int8, foundation.NonNull)
	narrow = checked(t, narrow, err)
	v, err := Factory.NewInteger(int64NonNullType, 1)
	v = checked(t, v, err)
	peer := scalarPeer{typ: narrow, integer: 1}
	if v.EqualsTo(peer) || peer.EqualsTo(v) {
		t.Fatal("equality discarded width")
	}
}
