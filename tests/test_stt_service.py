#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the audio handling shared by every STTService."""

import asyncio
from collections.abc import AsyncGenerator

import pytest

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    STTMetadataFrame,
)
from pipecat.services.stt_service import STTService
from pipecat.tests.utils import run_test

SAMPLE_RATE = 16000
# Distinct payloads so the audio handed to run_stt() shows which frame it came from.
INPUT_PCM = b"\x01\x00" * 160
OUTPUT_PCM = b"\x02\x00" * 160


def _make_capturing_service(**kwargs) -> STTService:
    """Build an STTService that records the audio handed to run_stt().

    Defined as a factory (not a module-level class) so this concrete subclass
    isn't picked up by the service-discovery scan in test_service_init.py.
    """

    class _CapturingSTTService(STTService):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.captured: list[bytes] = []

        async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
            self.captured.append(audio)
            return
            yield  # make this an async generator

    return _CapturingSTTService(**kwargs)


def _input_audio() -> InputAudioRawFrame:
    return InputAudioRawFrame(audio=INPUT_PCM, sample_rate=SAMPLE_RATE, num_channels=1)


def _output_audio() -> OutputAudioRawFrame:
    return OutputAudioRawFrame(audio=OUTPUT_PCM, sample_rate=SAMPLE_RATE, num_channels=1)


@pytest.mark.asyncio
async def test_output_audio_is_forwarded_without_being_transcribed():
    service = _make_capturing_service()

    await run_test(
        service,
        frames_to_send=[_input_audio(), _output_audio()],
        expected_down_frames=[STTMetadataFrame, InputAudioRawFrame, OutputAudioRawFrame],
    )

    assert service.captured == [INPUT_PCM]


@pytest.mark.asyncio
async def test_output_audio_is_forwarded_even_without_audio_passthrough():
    service = _make_capturing_service(audio_passthrough=False)

    await run_test(
        service,
        frames_to_send=[_input_audio(), _output_audio()],
        expected_down_frames=[STTMetadataFrame, OutputAudioRawFrame],
    )

    assert service.captured == [INPUT_PCM]


@pytest.mark.asyncio
async def test_reconnect_replays_buffer_in_order_with_live_frames_arriving_during_replay():
    from pipecat.frames.frames import InputAudioRawFrame
    from pipecat.processors.frame_processor import FrameDirection
    from pipecat.services.stt_service import STTService

    class _S(STTService):
        def __init__(self):
            super().__init__()
            self.order = []
            self.live = None

        async def run_stt(self, audio):
            yield None

        async def _process_audio_frame_unbuffered(self, frame, direction):
            self.order.append(frame.audio)
            if frame.audio == b"\x01" and self.live is not None:
                # A live frame arrives while the replay is mid-flight.
                live, self.live = self.live, None
                await self.process_audio_frame(live, direction)
            await asyncio.sleep(0)

    def f(b):
        return InputAudioRawFrame(audio=b, sample_rate=16000, num_channels=1)

    service = _S()

    async def fake_do_reconnect():
        service._reconnect_audio_buffer.extend(
            [(f(b"\x01"), FrameDirection.DOWNSTREAM), (f(b"\x02"), FrameDirection.DOWNSTREAM)]
        )

    service._do_reconnect = fake_do_reconnect
    service.live = f(b"\x03")

    await service._reconnect()

    assert service.order == [b"\x01", b"\x02", b"\x03"]
    assert service._reconnecting is False
    assert service._reconnect_audio_buffer == []
