package utils

import (
	"math"
	"testing"
)

func TestSessionTokenLifetime(t *testing.T) {
	first, ok := NextSessionToken()
	if !ok || first == 0 {
		t.Fatal("expected a nonzero token")
	}
	next, ok := NextSessionToken()
	if !ok || next <= first {
		t.Fatal("token not monotonically increasing")
	}
}

func TestSessionTokenExhaustion(t *testing.T) {
	var allocator sessionTokenAllocator
	allocator.last.Store(math.MaxUint64 - 1)
	last, ok := allocator.next()
	if !ok || uint64(last) != math.MaxUint64 {
		t.Fatal("last token should be usable")
	}
	for i := 0; i < 2; i++ {
		if token, ok := allocator.next(); ok || token != 0 {
			t.Fatal("exhausted allocator wrapped or reused a token")
		}
	}
}
