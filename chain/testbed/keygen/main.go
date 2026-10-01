// Command keygen generates N deterministic ECDSA keypairs for the testbed.
//
// TEST KEYS ONLY. Deterministic on purpose (reproducible genesis across
// machines/runs), never to be used for anything holding real value.
package main

import (
	"crypto/ecdsa"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/big"
	"os"

	"github.com/ethereum/go-ethereum/crypto"
)

type keyEntry struct {
	Address string `json:"address"`
	PrivKey string `json:"privkey"`
}

func main() {
	n := 6
	if len(os.Args) > 1 {
		fmt.Sscanf(os.Args[1], "%d", &n)
	}
	out := make([]keyEntry, n)
	for i := 0; i < n; i++ {
		// Deterministic scalar: i+1, as a 32-byte big-endian private key.
		// Valid secp256k1 scalars must be in [1, N-1]; i+1 for small i is safe.
		d := new(big.Int).SetInt64(int64(i + 1))
		priv := new(ecdsa.PrivateKey)
		priv.PublicKey.Curve = crypto.S256()
		priv.D = d
		priv.PublicKey.X, priv.PublicKey.Y = priv.PublicKey.Curve.ScalarBaseMult(d.Bytes())
		addr := crypto.PubkeyToAddress(priv.PublicKey)
		out[i] = keyEntry{
			Address: addr.Hex(),
			PrivKey: hex.EncodeToString(crypto.FromECDSA(priv)),
		}
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	enc.Encode(out)
}
