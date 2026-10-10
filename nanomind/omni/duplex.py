"""
nanomind/omni/duplex.py — Full-duplex conversational state machine and barge-in interruption detection.
"""
from enum import Enum
from typing import Dict, Any, Optional, List
import time


class DialogueState(str, Enum):
    """Dialogue agent states during streaming interaction."""
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"


class DuplexDialogueManager:
    """
    Manages simultaneous listening and speaking.
    Detects user barge-in (interruption while model is generating speech) and smoothly yields the floor.
    """

    def __init__(self, barge_in_sensitivity: float = 0.6, cooldown_ms: int = 400):
        self.state = DialogueState.LISTENING
        self.barge_in_sensitivity = barge_in_sensitivity
        self.cooldown_ms = cooldown_ms
        self.last_interruption_time: float = 0.0
        self.interruption_count: int = 0

    def on_user_speech_frame(self, user_speaking_prob: float) -> Dict[str, Any]:
        """
        Called when a new audio frame is captured from the user.
        user_speaking_prob: Voice Activity Detection (VAD) confidence in [0, 1].
        """
        now = time.time() * 1000.0  # ms
        interrupted = False

        if self.state == DialogueState.SPEAKING:
            if user_speaking_prob >= self.barge_in_sensitivity:
                if (now - self.last_interruption_time) > self.cooldown_ms:
                    self.state = DialogueState.INTERRUPTED
                    self.last_interruption_time = now
                    self.interruption_count += 1
                    interrupted = True

        elif self.state == DialogueState.INTERRUPTED:
            # Transition to listening to capture user's new question
            self.state = DialogueState.LISTENING

        return {
            "current_state": self.state.value,
            "interrupted": interrupted,
            "total_interruptions": self.interruption_count,
        }

    def start_speaking(self):
        """Model starts generating spoken output."""
        self.state = DialogueState.SPEAKING

    def finish_speaking(self):
        """Model completed current output."""
        if self.state == DialogueState.SPEAKING:
            self.state = DialogueState.LISTENING
