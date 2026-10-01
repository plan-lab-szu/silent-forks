# arm64 主机产出的 payload envelope（2026-09-18 归档）

这六个文件是**唯一不能在本机重新生成的东西**。其余跨机器证据都是日志，而日志里的结论可以靠这些 envelope 在任何一台机器上重新判决；envelope 本身只能由那台 arm64 主机产出。

原本它们只存在于 `xm-exchange/`（gitignored，属于每次可弃的传输目录）。为了让 arm64 主机之后不必再开机，在此归档一份。

## 来源主机

| | |
|---|---|
| 主机名 | `ttwang-surface` |
| 系统/架构 | `MINGW64_NT-10.0-26200-ARM64`，windows/**arm64** |
| Go | go1.26.7 |
| 核数 | 12 |
| 产出时间 | 2026-09-18（`differ-20260918` 与 `same-order-20260918` 两次交换） |

对端主机为 `HOME-DELL-DESKTOP`，windows/amd64，go1.26.5，16 核。

## 内容

见 `_table.md`（顺序、架构、post-state root、parent），校验和见 `SHA256SUMS`。

三个实验各两个累加顺序：

- `a__*` — A，高度调度的强制系统调用，浮点路径，空交易区块
- `ap__*` — A′，交易入口，浮点路径
- `b__*` — B，交易入口，规定路径

## 怎么在本机重新判决（不需要 arm64 主机）

在 `chain/geth/` 下，把 `_IN` 指向这里的文件。注意 build tag 要与**本机要扮演的顺序**一致，而不是与文件的顺序一致：

同一顺序、跨架构（`seq-fwd` 一致、`seq-rev` 不一致——用 observe 模式，因为两机差异不止顺序）：

```bash
DFRL_PAYLOAD_IN=<abs>/a__seq-fwd__ttwang-surface.json DFRL_XM_EXPECT=same-order go test -count=1 -v -tags e3lite_vulnerable -run TestDFRLTwoNodePayloadExchange ./eth/catalyst/
```

```bash
DFRL_PAYLOAD_IN=<abs>/a__seq-rev__ttwang-surface.json DFRL_XM_EXPECT=same-order-observe go test -count=1 -v -tags "e3lite_vulnerable e3lite_reverse" -run TestDFRLTwoNodePayloadExchange ./eth/catalyst/
```

不同顺序（本机 fwd 判 arm64 的 rev，预期 INVALID）：

```bash
DFRL_PAYLOAD_IN=<abs>/a__seq-rev__ttwang-surface.json go test -count=1 -v -tags e3lite_vulnerable -run TestDFRLTwoNodePayloadExchange ./eth/catalyst/
```

A′ 换 `DFRL_TXV_IN` + `TestDFRLTwoNodeTxPathDiverges`；B 换 `DFRL_DET_IN` + `TestDFRLTwoNodeDeterministicPathAgrees`。

判决前的守卫（传输完整性、parent 相同、交易表逐字节相同、顺序关系、同主机拒绝）照常全跑，所以拿归档文件重判和当时实时重判是同一件事。

## 结论与边界

`seq-fwd` is byte-identical on amd64 and arm64, whereas `seq-rev` differs.
Host stability is therefore a property of a particular evaluation order rather
than of the floating-point pipeline as a whole. The specified integer path
produces one root in all four cells.
