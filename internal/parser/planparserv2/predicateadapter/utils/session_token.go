// Package utils supplies shared predicate-adapter utilities.
package utils

import (
	"math"
	"sync/atomic"
)

// Process-local identity for one fixed schema view; zero is invalid.
// Tokens and their keys must never be persisted or sent over the wire.
type SessionToken uint64

// All sessions use this allocator, which must never be reset or replaced.
var sessionTokens sessionTokenAllocator

// Token allocator supporting isolated exhaustion tests; must not be copied after use.
type sessionTokenAllocator struct {
	last atomic.Uint64
}

// Reserves a unique token without wrapping; exhaustion is permanent.
// The caller decides how session creation handles exhaustion.
func (a *sessionTokenAllocator) next() (SessionToken, bool) {
	for {
		last := a.last.Load()
		if last == math.MaxUint64 {
			return 0, false
		}
		if a.last.CompareAndSwap(last, last+1) {
			return SessionToken(last + 1), true
		}
	}
}

// Reserves a process-local token from the shared allocator.
// Exhaustion returns (0, false); allocated tokens are never recycled.
func NextSessionToken() (SessionToken, bool) {
	return sessionTokens.next()
}
