#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import pytest

from pipecat.frames.frames import EndFrame, OutputAudioRawFrame, StartFrame
from pipecat.pipeline.parallel_pipeline import SEEN_IDS_MAX, ParallelPipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class _Pass(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


@pytest.mark.asyncio
async def test_seen_ids_and_frame_counter_stay_bounded():
    pp = ParallelPipeline([_Pass()], [_Pass()])
    pushed = []

    async def capture(frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append(frame)

    pp.push_frame = capture

    for _ in range(100_000):
        frame = OutputAudioRawFrame(audio=b"\x00\x00", sample_rate=8000, num_channels=1)
        await pp._parallel_push_frame(frame, FrameDirection.DOWNSTREAM)
        # Duplicate from the second branch is still deduplicated.
        await pp._parallel_push_frame(frame, FrameDirection.DOWNSTREAM)

    assert len(pushed) == 100_000
    assert len(pp._seen_ids) <= SEEN_IDS_MAX


@pytest.mark.asyncio
async def test_frame_counter_entry_removed_when_it_reaches_zero():
    pp = ParallelPipeline([_Pass()], [_Pass()])

    async def capture(frame, direction=FrameDirection.DOWNSTREAM):
        pass

    pp.push_frame = capture
    end = EndFrame()
    pp._frame_counter[end.id] = 2
    await pp._pipeline_sink_push_frame(end, FrameDirection.DOWNSTREAM)
    assert end.id in pp._frame_counter
    await pp._pipeline_sink_push_frame(end, FrameDirection.DOWNSTREAM)
    assert end.id not in pp._frame_counter
