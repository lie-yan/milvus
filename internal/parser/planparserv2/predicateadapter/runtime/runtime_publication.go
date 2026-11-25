package runtime

import "sync"

// Process-lifetime initialization state, acquired only when an adapter attempt starts.
var sharedRuntime runtimeLoader

// Returns one fully constructed process-wide runtime, caching either success or setup failure until restart.
func SharedRuntime() (*Runtime, error) {
	return sharedRuntime.load(NewRuntime)
}

// One-time publication owner; must not be copied after first use.
type runtimeLoader struct {
	// Serializes construction and publishes its outcome to all waiting readers.
	once sync.Once
	// Complete immutable bundle, absent when construction failed.
	runtime *Runtime
	// Original setup failure, retained without reclassification or retry.
	err error
}

// Runs the internal constructor once; concurrent callers wait for the same complete outcome.
func (l *runtimeLoader) load(build func() (*Runtime, error)) (*Runtime, error) {
	l.once.Do(func() {
		runtime, err := build()
		if err != nil {
			l.err = err
			return
		}
		l.runtime = runtime
	})
	return l.runtime, l.err
}
