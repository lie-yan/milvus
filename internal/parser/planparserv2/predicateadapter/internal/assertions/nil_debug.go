//go:build debug

// Package assertions guards internal programmer preconditions in debug builds.
package assertions

import "reflect"

// Rejects nil interfaces and typed nil values without invoking their methods.
func NotNil(value any) {
	if value == nil {
		panic("predicateadapter: expected non-nil value")
	}
	switch v := reflect.ValueOf(value); v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice, reflect.UnsafePointer:
		if v.IsNil() {
			panic("predicateadapter: expected non-nil value")
		}
	}
}

// Checks a statically known pointer without reflection or inspecting its target.
func NotNilPointer[T any](value *T) {
	if value == nil {
		panic("predicateadapter: expected non-nil pointer")
	}
}
