"""The stall rule of `stall_kicks`: forced pointer steps are counted, a step of the
model's own clears the count, and the restart rewinds the pointer to where the stall
began with the counters ahead of it cleared."""

from types import SimpleNamespace

import torch

from voxtream.utils.generator.context import FrameState
from voxtream.utils.generator.runtime import (
    restart_from_stall,
    update_indices_and_tokens,
)

CTX = SimpleNamespace(
    batch_size=2,
    phoneme_index_map={"0": [0, 1], "1": [0, 2], "2": [1, 1], "3": [1, 2]},
    config=SimpleNamespace(
        final_phone_hold_frames=12,
        frame_repeat_counter=3,
        num_phones_per_frame=2,
    ),
)
HOLD, STEP = torch.tensor(0), torch.tensor(2)
FRAME = torch.zeros(1, 4, dtype=torch.int64)


def initial(ptr: int = 10) -> FrameState:
    return FrameState(
        phone_emb_indices=torch.tensor([[[ptr, ptr]]] * 2),
        phone_emb_max_idx=ptr,
        eos_idx=10_000,
        state_counter={},
    )


def advance(state: FrameState, pred_shift: torch.Tensor, frames: int) -> FrameState:
    for idx in range(frames):
        _, state = update_indices_and_tokens(
            pred_shift,
            FRAME,
            idx,
            phone_seq_len=100,
            max_phone_idx=100,
            frame_state=state,
            ctx=CTX,
        )
    return state


def test_kicks_are_counted_and_a_step_clears_them():
    s = initial(10)
    s = advance(s, HOLD, 5)  # the 5th hold on one state exceeds the counter of 3
    assert s.phone_emb_max_idx == 11 and s.kicks == 1 and s.stall_anchor == 10
    s = advance(s, HOLD, 5)  # kicked again from the next row
    assert s.phone_emb_max_idx == 12 and s.kicks == 2 and s.stall_anchor == 10
    s = advance(s, STEP, 1)  # the model moves by itself
    assert s.phone_emb_max_idx == 13 and s.kicks == 0 and s.stall_anchor == -1
    s = advance(s, HOLD, 2)  # a short hold is not a kick
    assert s.kicks == 0 and s.phone_emb_max_idx == 13


def test_restart_rewinds_to_the_anchor():
    s = advance(initial(10), HOLD, 10)
    assert s.kicks == 2 and s.phone_emb_max_idx == 12
    r = restart_from_stall(s, CTX)
    assert r.phone_emb_max_idx == 10 and r.kicks == 0 and r.stall_anchor == -1
    assert r.phone_emb_indices.tolist() == [[[10, 10]]] * 2
    assert all(start < 10 for start, _ in r.state_counter)
    # after the restart the rows may be held again for the full counter
    r = advance(r, HOLD, 4)
    assert r.phone_emb_max_idx == 10 and r.kicks == 0
