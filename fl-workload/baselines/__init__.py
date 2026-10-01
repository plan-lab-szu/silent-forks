"""Comparison arms for the plain-FL workload (Phase 2).

Three arms, differing only in how client updates are combined:

  centralized      -- no federation at all; the reference for learning quality
  fedavg_offchain  -- federated, exact integer aggregation via dfrl-agg
  priorwork_float  -- federated, floating-point aggregation as the prior work did

The point of the first is U6/X4: if the federated integer arm tracks it, then the
chain is buying verifiability rather than accuracy, and the honest claim is about
overhead, not gains. The point of the third is C1: the same updates, aggregated
in float, put different bytes in different validators' state.
"""
