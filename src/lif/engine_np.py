"""CPU lane: the same tick in numpy, for hosts without Metal (x86 Linux).

This is phase-1's engine_naive.tick transcribed op-for-op into numpy, with one
optimisation: propagation gathers only the edges of spiking sources instead of
masking all E edges. Semantics that must not drift, because the parity gate
compares this lane against engine_fused bit-for-bit:

  * refractory refresh first, then the exact closed-form update (pre-decay g),
    both applied only where not refractory;
  * strict threshold v > V_TH gated by not-refractory;
  * arrivals read the delay ring slot BEFORE this tick's spike is stored;
  * contributions accumulate in exact integer arithmetic (int32 counts), then
    convert to float32 once -- np.bincount weighs in float64, which holds every
    int32 sum exactly, so the cast back to int32 reproduces the atomic int32
    adds of the kernel lanes;
  * synaptic input and external drive are shielded while refractory;
  * reset to V_0, g to 0, refractory reload from rfc_reload, then ring store.

float32 rounding: every array is float32 and every constant comes from
core.constants_f32() already rounded, and each product/sum is a separate numpy
op, so no contraction can merge rounds -- the same discipline the Metal lane
enforces with `#pragma clang fp contract(off)`.

Arguments chunk / use_async / edge_split / warmup are accepted for call-site
compatibility with engine_fused.run and ignored: there is nothing to compile,
partition, or pipeline here. cap is likewise accepted and unused; recording is
per-tick on the host and cannot overflow.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from lif import core

try:  # on Metal hosts reuse the canonical definition so isinstance checks hold
    from lif.engine_naive import RunResult
except Exception:  # noqa: BLE001 - no mlx: engine_naive imports mlx at top

    @dataclass
    class RunResult:  # type: ignore[no-redef]
        spike_counts: np.ndarray   # int32[N]
        v_final: np.ndarray        # float32[N]
        g_final: np.ndarray        # float32[N]
        n_ticks: int
        seconds: float             # measured wall clock
        peak_bytes: int
        events: np.ndarray | None = None  # int32[E, 2] of (tick, neuron)

        def counts_sha256(self) -> str:
            import hashlib
            a = np.ascontiguousarray(self.spike_counts.astype(np.int32))
            return hashlib.sha256(a.tobytes()).hexdigest()

        def total_spikes(self) -> int:
            return int(self.spike_counts.sum())


def _edges_of(row_ptr: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Concatenated CSR edge indices of `sources`, without a Python loop.

    For row lengths l_i and starts s_i, output index j of row i is
    s_i + (j - offset_i) = j + (s_i - offset_i); np.repeat broadcasts that
    per-row correction. `sources` must be sorted (np.flatnonzero guarantees it)
    so the produced edge order matches walking rows in source order.
    """
    starts = row_ptr[sources]
    lens = row_ptr[sources + 1] - starts
    total = int(lens.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    cum = np.cumsum(lens)
    return (np.arange(total, dtype=np.int64)
            + np.repeat(starts - (cum - lens), lens))


def run(pack: core.Pack, stim: core.Stimulus, silenced: np.ndarray | None = None,
        chunk: int = 32, use_async: bool = True, warmup: int = 50,
        edge_split: int = 1, record: bool = False,
        cap: int | None = None) -> RunResult:
    # Same contract as the other lanes; the parameters this lane ignores are
    # still refused when nonsense.
    if chunk < 1:
        raise ValueError(f"chunk must be at least 1, got {chunk}")
    if warmup < 0:
        raise ValueError(f"warmup must be 0 or more, got {warmup}")

    # Host copies whatever the pack flavour is (numpy on CPU hosts, mx on
    # Metal hosts so A/B parity runs can call this lane there too).
    row_ptr = np.asarray(pack.row_ptr)
    destinations = np.asarray(pack.destinations)
    edge_src = np.asarray(pack.edge_src)
    signed_counts = np.asarray(pack.signed_counts)
    draws = np.asarray(stim.draws)
    targets = np.asarray(stim.targets)

    # Silencing as data, as in engine_naive: counts of silenced sources zeroed
    # once per run. A silenced neuron still integrates, spikes and counts.
    mask = core.silenced_mask(pack, silenced)
    if mask is not None:
        signed_counts = np.where(np.asarray(mask)[edge_src],
                                 np.int32(0), signed_counts)

    cf = core.constants_f32()
    c_v0_term, c_couple_g, c_decay_v = cf["v0_term"], cf["couple_g"], cf["decay_v"]
    c_decay_g, c_v_th, c_w_syn = cf["decay_g"], cf["v_th"], cf["w_syn"]
    c_w_ext, c_v_0 = cf["w_ext"], cf["v_0"]

    N = pack.n_neurons
    n = stim.n_ticks

    # initial_state is numpy on CPU hosts and mx on Metal hosts; normalise.
    st = core.initial_state(pack, stim)
    v = np.asarray(st["v"], dtype=np.float32)
    g = np.asarray(st["g"], dtype=np.float32)
    rfc = np.asarray(st["rfc"], dtype=np.int32)
    rfc_reload = np.asarray(st["rfc_reload"], dtype=np.int32)
    counts = np.asarray(st["counts"], dtype=np.int32).copy()
    ring_np = np.asarray(st["ring"], dtype=np.bool_)
    ring = [np.ascontiguousarray(r) for r in ring_np]  # DELAY_TICKS rows

    f32 = np.float32
    events = [] if record else None
    start = time.perf_counter()
    for t in range(n):
        slot = t % core.DELAY_TICKS

        # --- 1. refractory refresh, then the exact closed-form update.
        rfc = np.maximum(rfc - 1, np.int32(0))
        not_ref = rfc == 0
        v_upd = c_v0_term + (g * c_couple_g + v * c_decay_v)
        g_upd = g * c_decay_g
        v = np.where(not_ref, v_upd, v)
        g = np.where(not_ref, g_upd, g)

        # --- 2. strict threshold, refractory neurons cannot fire.
        spike = not_ref & (v > c_v_th)

        # --- 3. spikes emitted DELAY_TICKS ago, read before overwrite.
        delayed = ring[slot]

        # --- 4. propagate: edges of spiking sources only, integer-exact.
        contrib = None
        sources = np.flatnonzero(delayed)
        if sources.size:
            idx = _edges_of(row_ptr, sources)
            if idx.size:
                w = signed_counts[idx].astype(np.float64)
                contrib = np.bincount(destinations[idx], weights=w,
                                      minlength=N).astype(np.int32)
        if contrib is not None:
            g = g + np.where(not_ref, contrib.astype(np.float32) * c_w_syn, f32(0.0))

        # External Poisson drive lands directly on v, shielded while
        # refractory; targets are unique so the scatter cannot collide.
        stim_row = draws[t]
        hit = targets[stim_row]
        if hit.size:
            v[hit] += np.where(not_ref[hit], c_w_ext, f32(0.0))

        # --- 5. reset, refractory reload, ring store.
        v = np.where(spike, c_v_0, v)
        g = np.where(spike, f32(0.0), g)
        rfc = np.where(spike, rfc_reload, rfc)
        ring[slot] = spike
        counts = counts + spike.astype(np.int32)

        if record:
            fired = np.flatnonzero(spike).astype(np.int32)
            events.append(np.column_stack(
                [np.full(fired.size, t, dtype=np.int32), fired]))
    elapsed = time.perf_counter() - start

    if record:
        events_arr = (np.concatenate(events) if events
                      else np.zeros((0, 2), dtype=np.int32))
    else:
        events_arr = None

    return RunResult(
        spike_counts=counts, v_final=v, g_final=g, n_ticks=n,
        seconds=elapsed, peak_bytes=0, events=events_arr,
    )
