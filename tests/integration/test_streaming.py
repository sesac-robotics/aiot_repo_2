"""청크 분할·손상·느린 구독자와 영상 상태 격리를 검증합니다."""

import asyncio

import pytest

from Server.AIFitService.adapters.frame_buffer import LatestFrameBuffer
from shared.mjpeg import JPEGFrameParser


@pytest.mark.parametrize("cut", range(1, 9))
def test_split_marker_and_multiple_frames(cut):
    first = b"\xff\xd8first\xff\xd9"
    second = b"\xff\xd8second\xff\xd9"
    wire = b"header\r\n" + first + b"boundary" + second
    parser = JPEGFrameParser(32)
    frames = []
    for start in range(0, len(wire), cut):
        frames.extend(parser.feed(wire[start : start + cut]))
        assert parser.buffered_bytes <= 32
    assert frames == [first, second]


def test_oversized_partial_frame_does_not_grow_and_recovers():
    parser = JPEGFrameParser(16)
    assert parser.feed(b"\xff\xd8" + b"x" * 1000) == []
    assert parser.buffered_bytes <= 16
    assert parser.feed(b"\xff\xd8valid\xff\xd9") == [b"\xff\xd8valid\xff\xd9"]
    assert parser.feed(b"\xff\xd8" + b"x" * 50 + b"\xff\xd9") == []


def test_buffers_are_app_local_and_only_keep_latest_frame():
    async def scenario():
        first = LatestFrameBuffer()
        second = LatestFrameBuffer()
        assert (await second.status())["has_frame"] is False
        await first.publish(b"old")
        await first.publish(b"latest")
        assert await first.next_frame(0) == (2, b"latest")
        assert (await second.status())["sequence"] == 0
        waiting = asyncio.create_task(first.next_frame(2))
        await asyncio.sleep(0)
        assert not waiting.done()
        await first.publish(b"new")
        assert await waiting == (3, b"new")

    asyncio.run(scenario())
