"""`prompt_bias_start` of the sink attention: the prompt-key bias applies to a row only
once the stream has generated that many frames, by the row's own time in the stream, so
the opening is attended as trained and a cache rebuild re-encodes a row as it was first
generated. The visible keys never change."""

import torch

from voxtream.utils.sink_attention import SinkAttention, SinkAttentionConfig

PROMPT_LEN, MAX_SEQ_LEN, BIAS = 3, 32, 3.0
CAUSAL = torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN, dtype=torch.bool))


def sink(start: int, bias: float = BIAS) -> SinkAttention:
    s = SinkAttention(
        SinkAttentionConfig(
            audio_window_size=16,
            prompt_bias=bias,
            prompt_bias_dtype=torch.float32,
            prompt_bias_start=start,
        )
    )
    s._sink_prompt_len = PROMPT_LEN
    return s


def additive(mask: torch.Tensor) -> torch.Tensor:
    return torch.zeros(mask.shape).masked_fill(~mask, float("-inf"))


def test_no_bias_before_start_and_bias_after():
    plain = sink(start=0, bias=None)
    for generated in range(0, 10):
        pos = torch.tensor([[PROMPT_LEN + generated]])
        visible = plain._get_temp_former_sink_mask(pos, CAUSAL)
        got = sink(start=4)._get_temp_former_sink_mask(pos, CAUSAL, global_pos=pos)
        expected = additive(visible)
        if generated >= 4:
            expected[..., :PROMPT_LEN] = BIAS
        assert torch.equal(got, expected), generated


def test_start_zero_is_the_bias_everywhere():
    pos = torch.tensor([[PROMPT_LEN]])
    always = sink(start=0)._get_temp_former_sink_mask(pos, CAUSAL, global_pos=pos)
    assert torch.all(always[..., :PROMPT_LEN] == BIAS)


def test_global_position_defaults_to_the_window_offset():
    s = sink(start=4)
    s._sink_pos_offset = 10  # a rebuilt window: local positions lag the stream by 10
    local = torch.tensor([[PROMPT_LEN + 1]])  # stream position PROMPT_LEN + 11: past
    assert torch.all(
        s._get_temp_former_sink_mask(local, CAUSAL)[..., :PROMPT_LEN] == BIAS
    )
    s._sink_pos_offset = 0  # stream position PROMPT_LEN + 1: before
    assert torch.all(s._get_temp_former_sink_mask(local, CAUSAL)[..., :PROMPT_LEN] == 0)


def test_rebuild_rows_take_the_bias_by_their_own_time():
    # a rebuild re-encodes the prompt and a tail at local positions 0..7, which lie at
    # stream positions 2..9 (offset 2): rows generated before frame 4 of the stream
    # (stream position < PROMPT_LEN + 4 = 7) see no bias, the later ones do
    local = torch.arange(8).unsqueeze(0)
    got = sink(start=4)._get_temp_former_sink_mask(local, CAUSAL, global_pos=local + 2)
    on_prompt = got[0, :, :PROMPT_LEN]
    assert torch.all(on_prompt[:5] <= 0), "stream positions 2..6 are before the start"
    assert torch.all(on_prompt[5:] == BIAS), "stream positions 7..9 are after it"
    # the prompt rows see only prompt keys whatever the bias
    assert torch.all(torch.isinf(got[0, :PROMPT_LEN, PROMPT_LEN:]))
