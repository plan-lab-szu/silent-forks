// Command dfrl-agg exposes the deterministic aggregation primitives to
// off-chain callers.
//
// The point of shipping this as a CLI is that the off-chain aggregator, the
// geth precompile and the fused opcode all call the SAME package. That makes
// U6 (on-chain and off-chain results agree) and X6 (validators agree byte for
// byte) properties of the build rather than things to be reconciled later.
package main

import (
	"encoding/hex"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	dfrlagg "dfrl-agg"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "fedavg":
		err = cmdAgg(os.Args[2:], false)
	case "trimmed":
		err = cmdAgg(os.Args[2:], true)
	case "diverge":
		err = cmdDiverge(os.Args[2:])
	case "root":
		err = cmdRoot(os.Args[2:])
	case "-h", "--help", "help":
		usage()
	default:
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "dfrl-agg:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `usage: dfrl-agg <command> [flags]

  fedavg   -w s1,s2,...  [-o out.bin]  in1.bin in2.bin ...
           weighted average of canonically encoded tensors

  trimmed  -k K [-o out.bin]  in1.bin in2.bin ...
           coordinate-wise trimmed mean

  root     file.bin
           print the state root of a canonically encoded tensor

  diverge  [-n N] [-p P] [-seed S]
           float32 aggregation under several conforming strategies,
           printing one state root per strategy (C1 negative control;
           never on the consensus path)
`)
}

func loadTensors(paths []string) ([]dfrlagg.Tensor, error) {
	if len(paths) == 0 {
		return nil, fmt.Errorf("no input files")
	}
	out := make([]dfrlagg.Tensor, 0, len(paths))
	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err != nil {
			return nil, err
		}
		t, err := dfrlagg.Decode(b)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", p, err)
		}
		out = append(out, t)
	}
	return out, nil
}

// synthAddr assigns distinct positional identities when an off-chain caller
// has no account addresses. Identities support duplicate-membership checks;
// coordinate-wise TrimmedMean orders only numerical values.
func synthAddr(i int) dfrlagg.Addr {
	var a dfrlagg.Addr
	a[19] = byte(i + 1)
	a[18] = byte((i + 1) >> 8)
	return a
}

func cmdAgg(args []string, trimmed bool) error {
	fs := flag.NewFlagSet("agg", flag.ContinueOnError)
	weights := fs.String("w", "", "comma-separated integer weights (fedavg only)")
	k := fs.Int("k", 1, "number of values trimmed from each end (trimmed only)")
	out := fs.String("o", "", "write the canonical encoding of the result here")
	if err := fs.Parse(args); err != nil {
		return err
	}
	tensors, err := loadTensors(fs.Args())
	if err != nil {
		return err
	}

	ws := make([]int64, len(tensors))
	if trimmed {
		for i := range ws {
			ws[i] = 1 // trimming is deliberately unweighted
		}
	} else {
		parts := strings.Split(*weights, ",")
		if len(parts) != len(tensors) {
			return fmt.Errorf("got %d weights for %d tensors", len(parts), len(tensors))
		}
		for i, s := range parts {
			v, err := strconv.ParseInt(strings.TrimSpace(s), 10, 64)
			if err != nil {
				return fmt.Errorf("weight %q: %w", s, err)
			}
			ws[i] = v
		}
	}

	ups := make([]dfrlagg.Update, len(tensors))
	for i := range tensors {
		ups[i] = dfrlagg.Update{Addr: synthAddr(i), Weight: ws[i], Tensor: tensors[i]}
	}

	var res dfrlagg.Tensor
	if trimmed {
		res, err = dfrlagg.TrimmedMean(ups, *k)
	} else {
		res, err = dfrlagg.FedAvg(ups)
	}
	if err != nil {
		return err
	}
	root, err := dfrlagg.StateRoot(res)
	if err != nil {
		return err
	}
	if *out != "" {
		enc, err := dfrlagg.Encode(res)
		if err != nil {
			return err
		}
		if err := os.WriteFile(*out, enc, 0o644); err != nil {
			return err
		}
	}
	fmt.Printf("%s\n", hex.EncodeToString(root[:]))
	return nil
}

func cmdRoot(args []string) error {
	tensors, err := loadTensors(args)
	if err != nil {
		return err
	}
	for i, t := range tensors {
		r, err := dfrlagg.StateRoot(t)
		if err != nil {
			return err
		}
		fmt.Printf("%s  %s\n", hex.EncodeToString(r[:]), args[i])
	}
	return nil
}

func cmdDiverge(args []string) error {
	fs := flag.NewFlagSet("diverge", flag.ContinueOnError)
	n := fs.Int("n", 10, "number of clients")
	p := fs.Int("p", 250000, "parameters per client")
	seed := fs.Uint64("seed", 20260823, "input seed")
	if err := fs.Parse(args); err != nil {
		return err
	}
	theta := make([][]float32, *n)
	weights := make([]float32, *n)
	for i := 0; i < *n; i++ {
		theta[i] = dfrlagg.GenFloat32(*seed+uint64(i)*0x1000, *p)
		weights[i] = float32(dfrlagg.PaperWeights[i%len(dfrlagg.PaperWeights)])
	}

	base := dfrlagg.FloatAvg(theta, weights, dfrlagg.SeqFwd)
	roots := map[string]int{}
	fmt.Printf("%s/%s | N=%d P=%d seed=%d\n", goos(), goarch(), *n, *p, *seed)
	fmt.Printf("%-14s %-18s %12s %12s\n", "strategy", "root[:16]", "differs", "max |diff|")
	fmt.Println(strings.Repeat("-", 60))
	for _, st := range dfrlagg.AllFloatStrategies {
		v := dfrlagg.FloatAvg(theta, weights, st)
		diff, maxAbs := compare(v, base)
		r := dfrlagg.FloatRoot(v)
		h := hex.EncodeToString(r[:])
		roots[h]++
		fmt.Printf("%-14s %-18s %12d %12.3e\n", st, h[:16], diff, maxAbs)
	}
	fmt.Println(strings.Repeat("-", 60))
	fmt.Printf("%d strategies -> %d distinct state roots\n", len(dfrlagg.AllFloatStrategies), len(roots))

	// The integer path on the same inputs, for contrast.
	ups := make([]dfrlagg.Update, *n)
	for i := 0; i < *n; i++ {
		t, err := dfrlagg.Quantize(theta[i], []uint32{uint32(*p)}, dfrlagg.DefaultF)
		if err != nil {
			return err
		}
		ups[i] = dfrlagg.Update{Addr: synthAddr(i), Weight: dfrlagg.PaperWeights[i%len(dfrlagg.PaperWeights)], Tensor: t}
	}
	agg, err := dfrlagg.FedAvg(ups)
	if err != nil {
		return err
	}
	ir, err := dfrlagg.StateRoot(agg)
	if err != nil {
		return err
	}
	fmt.Printf("integer path (F=%d): single root %s\n", dfrlagg.DefaultF, hex.EncodeToString(ir[:])[:16])
	return nil
}

func compare(a, b []float32) (int, float64) {
	diff := 0
	var maxAbs float64
	for i := range a {
		if a[i] != b[i] {
			diff++
			if d := absf(float64(a[i]) - float64(b[i])); d > maxAbs {
				maxAbs = d
			}
		}
	}
	return diff, maxAbs
}

func absf(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
