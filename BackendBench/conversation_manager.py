# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Conversation Manager for LLM iterative refinement with history tracking.

This module provides conversation history management for LLM-based kernel generation,
allowing the LLM to learn from previous attempts and build context across multiple
refinement rounds.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class ConversationTurn:
    """Represents one turn in the conversation between user and LLM."""
    prompt: str
    response: str
    feedback: Optional[str]
    attempt_number: int
    timestamp: datetime
    success: bool
    feedback_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """Create from dictionary loaded from JSON."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class ConversationManager:
    """Manages conversation history for iterative LLM refinement."""

    def __init__(self, op_name: str, debug_mode: bool = False):
        """
        Initialize conversation manager.

        Args:
            op_name: Name of the operation (e.g., 'add', 'mul', 'relu')
            debug_mode: Whether to enable detailed logging
        """
        self.op_name = op_name
        self.debug_mode = debug_mode
        self.conversation_history: List[ConversationTurn] = []
        self.initial_prompt: Optional[str] = None
        self.framework: str = "triton"
        self.max_attempts: int = 5
        self.start_timestamp: datetime = datetime.now()

    def start_conversation(self, initial_prompt: str, framework: str = "triton", max_attempts: int = 5) -> str:
        """
        Start a new conversation with the initial prompt.

        Args:
            initial_prompt: The first prompt to send to the LLM
            framework: Framework being used (triton, pytorch, etc.)
            max_attempts: Maximum number of attempts allowed

        Returns:
            The initial prompt (for first attempt)
        """
        self.initial_prompt = initial_prompt
        self.framework = framework
        self.max_attempts = max_attempts
        self.start_timestamp = datetime.now()

        if self.debug_mode:
            print(f"🎯 Starting conversation for {self.op_name} (framework: {framework}, max_attempts: {max_attempts})")

        return initial_prompt

    def add_response(self, response: str, attempt: int, prompt: str = "") -> None:
        """
        Add LLM response to the current turn.

        Args:
            response: The LLM's generated response
            attempt: Current attempt number (1-based)
            prompt: The actual prompt that was sent to the LLM
        """
        # Create new turn or update existing one
        current_turn = None
        for turn in self.conversation_history:
            if turn.attempt_number == attempt:
                current_turn = turn
                break

        if current_turn is None:
            # Create new turn
            current_turn = ConversationTurn(
                prompt=prompt,  # Store the actual prompt that was sent
                response=response,
                feedback=None,
                attempt_number=attempt,
                timestamp=datetime.now(),
                success=False,
                feedback_info=None
            )
            self.conversation_history.append(current_turn)
        else:
            # Update existing turn
            current_turn.prompt = prompt  # Update with actual prompt
            current_turn.response = response
            current_turn.timestamp = datetime.now()

        if self.debug_mode:
            print(f"📝 Added response for {self.op_name} attempt {attempt} ({len(response)} chars)")

    def add_feedback(self, feedback_info: Dict[str, Any], attempt: int) -> None:
        """
        Add feedback to the current turn.

        Args:
            feedback_info: Dictionary containing feedback details
            attempt: Current attempt number (1-based)
        """
        # Find the turn for this attempt
        current_turn = None
        for turn in self.conversation_history:
            if turn.attempt_number == attempt:
                current_turn = turn
                break

        if current_turn is None:
            print(f"⚠️ Warning: No turn found for attempt {attempt} when adding feedback")
            return

        # Format feedback text
        feedback_text = self._format_feedback_text(feedback_info)

        # Update turn with feedback
        current_turn.feedback = feedback_text
        current_turn.feedback_info = feedback_info
        current_turn.success = feedback_info.get('success', False)

        if self.debug_mode:
            success_status = "✅ SUCCESS" if current_turn.success else "❌ FAILED"
            print(f"🔄 Added feedback for {self.op_name} attempt {attempt}: {success_status}")

    def _format_feedback_text(self, feedback_info: Dict[str, Any]) -> str:
        """Format feedback information into readable text."""
        feedback_parts = []

        if feedback_info.get("compilation_error"):
            feedback_parts.append(f"COMPILATION ERROR:\n{feedback_info['compilation_error']}")

        if feedback_info.get("test_errors"):
            feedback_parts.append("TEST ERRORS:")
            for i, error in enumerate(feedback_info["test_errors"]):
                feedback_parts.append(f"\nTest Case {i + 1}:")
                feedback_parts.append(f"Input: {error.get('test_input', 'N/A')}")
                feedback_parts.append(f"Error: {error.get('error', 'N/A')}")
                if error.get('traceback'):
                    feedback_parts.append(f"Traceback:\n{error['traceback']}")

        if feedback_info.get("summary"):
            feedback_parts.append(f"Summary: {feedback_info['summary']}")

        return "\n".join(feedback_parts) if feedback_parts else "No specific feedback available"

    def build_next_prompt(self, template_manager, op_signature: str, op_description: str) -> str:
        """
        Build the next prompt with full conversation history.

        Args:
            template_manager: KernelTemplateManager instance for prompt formatting
            op_signature: Operation signature string
            op_description: Operation description string

        Returns:
            Complete prompt with conversation history
        """
        if not self.conversation_history:
            # First attempt - return initial prompt
            if self.initial_prompt is None:
                raise ValueError("Initial prompt not set. Call start_conversation() first.")
            return self.initial_prompt

        # Build conversation history context - just use the existing ConversationTurn objects
        template_turns = self.conversation_history

        # Use template manager to create conversation prompt
        if hasattr(template_manager, 'create_conversation_refinement_prompt'):
            return template_manager.create_conversation_refinement_prompt(
                initial_prompt=self.initial_prompt,
                conversation_history=template_turns,
                op_name=self.op_name,
                op_signature=op_signature,
                op_description=op_description,
                framework=self.framework
            )
        else:
            # Fallback to basic conversation formatting
            return self._build_basic_conversation_prompt()

    def _build_basic_conversation_prompt(self) -> str:
        """Build basic conversation prompt without template manager."""
        if self.initial_prompt is None:
            raise ValueError("Initial prompt not set. Call start_conversation() first.")

        prompt_parts = [self.initial_prompt]

        for turn in self.conversation_history:
            prompt_parts.append(f"\n## ATTEMPT {turn.attempt_number}")
            prompt_parts.append(f"You generated:\n```python\n{turn.response}\n```")
            if turn.feedback:
                prompt_parts.append(f"Feedback:\n{turn.feedback}")

        next_attempt = len(self.conversation_history) + 1
        prompt_parts.append(f"\n## ATTEMPT {next_attempt}")
        prompt_parts.append("Please generate an improved version based on the conversation above.")

        return "\n".join(prompt_parts)

    def get_debug_log(self) -> str:
        """Return full conversation formatted for debugging."""
        log_parts = [
            f"=== CONVERSATION LOG: {self.op_name} ===",
            f"Timestamp: {self.start_timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Max Attempts: {self.max_attempts}",
            f"Framework: {self.framework}",
            ""
        ]

        if self.initial_prompt:
            log_parts.extend([
                "--- INITIAL PROMPT ---",
                self.initial_prompt,
                ""
            ])

        for turn in self.conversation_history:
            log_parts.extend([
                f"--- ATTEMPT {turn.attempt_number} ---",
                f"Timestamp: {turn.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                f"Response:",
                turn.response,
                ""
            ])

            if turn.feedback:
                log_parts.extend([
                    f"Feedback:",
                    turn.feedback,
                    ""
                ])

        # Final result summary
        final_result = self.conversation_history[-1] if self.conversation_history else None
        if final_result:
            log_parts.extend([
                "--- FINAL RESULT ---",
                f"Success: {final_result.success}",
                f"Total Attempts: {len(self.conversation_history)}",
                ""
            ])

        return "\n".join(log_parts)

    def save_conversation_log(self, log_dir: str) -> None:
        """
        Save detailed conversation log to file.

        Args:
            log_dir: Directory to save the log file
        """
        os.makedirs(log_dir, exist_ok=True)

        # Save human-readable log
        log_file = os.path.join(log_dir, f"{self.op_name}_conversation.log")
        with open(log_file, "w") as f:
            f.write(self.get_debug_log())

        # Save machine-readable JSON
        json_file = os.path.join(log_dir, f"{self.op_name}_conversation.json")
        conversation_data = {
            "op_name": self.op_name,
            "framework": self.framework,
            "max_attempts": self.max_attempts,
            "start_timestamp": self.start_timestamp.isoformat(),
            "initial_prompt": self.initial_prompt,
            "conversation_history": [turn.to_dict() for turn in self.conversation_history],
            "summary": self.get_conversation_summary()
        }

        with open(json_file, "w") as f:
            json.dump(conversation_data, f, indent=2)

        if self.debug_mode:
            print(f"💾 Saved conversation logs for {self.op_name} to {log_dir}")

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary statistics for this conversation."""
        total_attempts = len(self.conversation_history)
        successful_attempts = sum(1 for turn in self.conversation_history if turn.success)

        summary = {
            "op_name": self.op_name,
            "framework": self.framework,
            "total_attempts": total_attempts,
            "successful_attempts": successful_attempts,
            "final_success": self.conversation_history[-1].success if self.conversation_history else False,
            "conversation_length_chars": sum(len(turn.response) + len(turn.feedback or "") for turn in self.conversation_history),
            "start_timestamp": self.start_timestamp.isoformat(),
            "duration_seconds": (datetime.now() - self.start_timestamp).total_seconds() if self.conversation_history else 0
        }

        return summary

    def is_conversation_active(self) -> bool:
        """Check if conversation is still active (not finished)."""
        if not self.conversation_history:
            return True  # Not started yet

        # Check if last attempt was successful
        last_turn = self.conversation_history[-1]
        if last_turn.success:
            return False  # Successfully completed

        # Check if max attempts reached
        if len(self.conversation_history) >= self.max_attempts:
            return False  # Max attempts reached

        return True  # Still active

    def get_next_attempt_number(self) -> int:
        """Get the attempt number for the next turn."""
        return len(self.conversation_history) + 1
