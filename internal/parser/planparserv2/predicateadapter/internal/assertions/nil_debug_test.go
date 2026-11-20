//go:build debug

package assertions

import (
	"testing"
	"unsafe"

	"github.com/stretchr/testify/require"
)

func TestNilGuards(t *testing.T) {
	for _, value := range []any{nil, (*int)(nil), []int(nil), map[string]int(nil), (func())(nil), (chan int)(nil), unsafe.Pointer(nil)} {
		require.Panics(t, func() { NotNil(value) })
	}
	for _, value := range []any{0, "", []int{}, map[string]int{}, new(int)} {
		require.NotPanics(t, func() { NotNil(value) })
	}
	require.Panics(t, func() { NotNilPointer[int](nil) })
	var target any
	// A non-nil pointer remains valid even when its target contains nil.
	require.NotPanics(t, func() { NotNilPointer(&target) })
}
