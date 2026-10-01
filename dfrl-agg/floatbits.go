package dfrlagg

import "math"

// float32bits is isolated here so that floatref.go stays free of anything that
// could be mistaken for a consensus-path helper.
func float32bits(f float32) uint32 { return math.Float32bits(f) }
