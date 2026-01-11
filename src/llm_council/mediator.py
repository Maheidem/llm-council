"""Mediator persona role for LLM Council.

The mediator controls discussion flow, decides when to vote,
and synthesizes proposals - but does NOT interpret vote results.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from .models import Persona, Message
from .discussion import DiscussionState, DiscussionPhase

logger = logging.getLogger(__name__)


@dataclass
class MediatorDirective:
    """A directive from the mediator."""
    action: str  # "continue", "call_vote", "synthesize", "close"
    proposal: Optional[str] = None
    target_personas: Optional[list[str]] = None
    reason: Optional[str] = None


class MediatorRole:
    """Manages mediator persona behavior and prompts.

    The mediator:
    - Goes first in each round
    - Summarizes progress and steers discussion
    - Decides when to call a vote
    - Synthesizes proposals for voting
    - Does NOT interpret or count votes (VotingMachine does that)
    """

    MEDIATOR_SYSTEM_PROMPT_ADDON = """
You are the designated MEDIATOR of this council discussion.

Your responsibilities:
1. Guide the discussion toward the objective
2. Summarize key points and areas of agreement/disagreement
3. Ensure all perspectives are heard
4. Call for a vote when appropriate using [CALL_VOTE]
5. Synthesize proposals that capture the emerging consensus

When calling for a vote, use this format:
[ACTION] call_vote
[PROPOSAL] <your synthesized proposal for the group to vote on>
[REASONING] <why you believe it's time to vote>

You go FIRST in each round. Keep your contributions focused on facilitation.
Do NOT vote yourself - you are neutral."""

    def __init__(self, persona: Persona, persona_index: int = 0):
        """Initialize mediator role.

        Args:
            persona: The persona assigned as mediator
            persona_index: Index in the personas list (for ordering)
        """
        self.persona = persona
        self.persona_index = persona_index

    def get_system_prompt(self) -> str:
        """Get enhanced system prompt for mediator."""
        base_prompt = self.persona.to_system_prompt()
        return f"{base_prompt}\n{self.MEDIATOR_SYSTEM_PROMPT_ADDON}"

    def get_discussion_prompt(
        self,
        round_num: int,
        topic: str,
        objective: str,
        history_text: str,
        state: DiscussionState,
    ) -> str:
        """Generate mediator-specific discussion prompt.

        Args:
            round_num: Current round number
            topic: Discussion topic
            objective: Discussion objective
            history_text: Formatted discussion history
            state: Current discussion state

        Returns:
            Prompt string for mediator
        """
        parts = [
            f"TOPIC: {topic}",
            f"OBJECTIVE: {objective}",
            f"CURRENT ROUND: {round_num}",
            f"DISCUSSION PHASE: {state.phase.value}",
        ]

        if history_text:
            parts.append(f"\nPREVIOUS DISCUSSION:\n{history_text}")

        # Add phase-specific guidance
        if state.phase == DiscussionPhase.OPENING:
            parts.append("""
As mediator, open this round by:
1. Briefly stating the topic and what we aim to decide
2. Inviting each council member to share their initial perspective
3. Setting expectations for constructive dialogue""")

        elif state.phase == DiscussionPhase.DELIBERATION:
            parts.append(f"""
As mediator, guide the deliberation:
1. Summarize the key positions that have emerged
2. Identify areas of agreement and remaining disagreements
3. Ask targeted questions to clarify or bridge differences
4. Passes this round: {state.total_passes} / Contributions: {state.total_contributions}""")

        elif state.phase == DiscussionPhase.SYNTHESIS:
            parts.append("""
As mediator, work toward synthesis:
1. Identify the emerging consensus (if any)
2. Propose a synthesized position that addresses key concerns
3. If ready, use [CALL_VOTE] to initiate formal voting
4. If not ready, identify what remains to be resolved""")

        if state.should_auto_vote(5):  # Threshold for auto-vote consideration
            parts.append("""
NOTE: Multiple personas have passed recently. Consider whether the group
is ready to vote or if further discussion is needed.""")

        return "\n\n".join(parts)

    @staticmethod
    def create_mediator_from_persona(
        persona: Persona,
        is_mediator: bool = True
    ) -> Persona:
        """Create a mediator-enhanced version of a persona.

        Args:
            persona: Base persona to enhance
            is_mediator: Whether to apply mediator enhancements

        Returns:
            Enhanced persona (or original if not mediator)
        """
        if not is_mediator:
            return persona

        # Add mediator role to persona
        return Persona(
            name=persona.name,
            role=f"Mediator & {persona.role}",
            expertise=persona.expertise + ["facilitation", "conflict resolution"],
            personality_traits=persona.personality_traits + ["neutral", "organized"],
            perspective=f"As mediator: Guide discussion toward the objective while maintaining neutrality. {persona.perspective}",
            provider_config=persona.provider_config,
        )


def select_mediator(personas: list[Persona], index: int = 0) -> tuple[Persona, int]:
    """Select and configure the mediator from the persona list.

    By default, the first persona (index 0) becomes the mediator.
    The Diplomat persona is often a good choice if present.

    Args:
        personas: List of all personas
        index: Index of persona to make mediator (default: 0)

    Returns:
        Tuple of (mediator persona, mediator index)
    """
    if not personas:
        raise ValueError("Cannot select mediator from empty persona list")

    if index < 0 or index >= len(personas):
        logger.warning(f"Invalid mediator index {index}, using 0")
        index = 0

    # Look for "The Diplomat" as natural mediator
    for i, p in enumerate(personas):
        if "diplomat" in p.name.lower() or "diplomat" in p.role.lower():
            logger.info(f"Auto-selecting '{p.name}' as mediator (diplomat role)")
            index = i
            break

    mediator = MediatorRole.create_mediator_from_persona(personas[index], is_mediator=True)
    logger.info(f"Selected '{mediator.name}' as mediator (index {index})")

    return mediator, index


def reorder_personas_mediator_first(
    personas: list[Persona],
    mediator_index: int
) -> list[Persona]:
    """Reorder personas so mediator speaks first.

    Args:
        personas: Original persona list
        mediator_index: Index of the mediator

    Returns:
        Reordered list with mediator first
    """
    if mediator_index == 0:
        return personas

    mediator = personas[mediator_index]
    others = [p for i, p in enumerate(personas) if i != mediator_index]

    return [mediator] + others
