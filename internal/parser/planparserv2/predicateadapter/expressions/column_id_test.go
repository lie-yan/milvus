package expressions

import (
	"testing"

	expressioninterfaces "github.com/lie-yan/predicate-optimization/expressions/interfaces"
)

// Non-comparable identity representation with separate diagnostic spelling.
type columnIDPeer struct {
	parts []string
}

func (id columnIDPeer) Key() string    { return id.parts[0] }
func (id columnIDPeer) String() string { return id.parts[1] }
func (id columnIDPeer) EqualsTo(other expressioninterfaces.ColumnID) bool {
	return id.Key() == other.Key()
}

func TestColumnIdentity(t *testing.T) {
	a := NewColumnID(42, 101)
	b := NewColumnID(42, 101)
	peer := columnIDPeer{parts: []string{"mvcol:v1:42:d:101", "different display name"}}
	if !a.EqualsTo(b) || !a.EqualsTo(peer) {
		t.Fatal("equal keys must compare equal across representations")
	}
	for _, different := range []expressioninterfaces.ColumnID{
		NewColumnID(43, 101), NewColumnID(42, 102),
		columnIDPeer{parts: []string{"another-domain:42:101", a.String()}},
	} {
		if a.EqualsTo(different) {
			t.Fatal("different identity domains/fields/sessions compare equal")
		}
	}
}
