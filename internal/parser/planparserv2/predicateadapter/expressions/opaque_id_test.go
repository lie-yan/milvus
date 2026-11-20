package expressions

import (
	"testing"

	expressioninterfaces "github.com/lie-yan/predicate-optimization/expressions/interfaces"
	"github.com/stretchr/testify/require"
)

// Non-comparable representation with diagnostic spelling independent of identity.
type opaqueIDPeer struct{ parts []string }

func (id opaqueIDPeer) Key() string    { return id.parts[0] }
func (id opaqueIDPeer) String() string { return id.parts[1] }
func (id opaqueIDPeer) EqualsTo(other expressioninterfaces.OpaqueID) bool {
	return id.Key() == other.Key()
}

func TestOpaqueIdentity(t *testing.T) {
	id := NewOpaqueID(42, 0)
	peer := opaqueIDPeer{parts: []string{"mvopaque:v1:42:0", "preserved predicate"}}
	require.True(t, id.EqualsTo(peer))
	require.True(t, peer.EqualsTo(id))
	for _, other := range []expressioninterfaces.OpaqueID{
		NewOpaqueID(43, 0), NewOpaqueID(42, 1),
		opaqueIDPeer{parts: []string{"other-domain:42:0", id.String()}},
	} {
		require.False(t, id.EqualsTo(other))
	}
}
