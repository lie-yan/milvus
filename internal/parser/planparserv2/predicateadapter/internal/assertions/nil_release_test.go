//go:build !debug

package assertions

import "testing"

func TestNilGuardsDisabled(t *testing.T) {
	NotNil(nil)
	NotNil((*int)(nil))
	NotNilPointer[int](nil)
}
