"""Core council discussion engine."""

import json
from typing import Optional, Union

from .models import (
    Persona,
    Message,
    Vote,
    VoteChoice,
    RoundResult,
    CouncilSession,
    ConsensusType,
)
from .providers import LLMProvider, ProviderRegistry, create_provider


class CouncilEngine:
    """Engine for running council discussions."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        provider_registry: Optional[ProviderRegistry] = None,
        consensus_type: ConsensusType = ConsensusType.MAJORITY,
        max_rounds: int = 5,
        stalemate_threshold: int = 2,
    ):
        """Initialize the council engine.

        Args:
            provider: Single LLM provider for all responses (legacy mode)
            provider_registry: Registry for per-persona provider resolution
            consensus_type: Type of consensus required
            max_rounds: Maximum discussion rounds before forcing vote
            stalemate_threshold: Rounds without progress before calling stalemate

        Note: Either provider or provider_registry must be provided.
              If both are provided, provider_registry takes precedence for persona lookups,
              but the single provider is used as fallback.
        """
        self.provider = provider
        self.provider_registry = provider_registry
        self.consensus_type = consensus_type
        self.max_rounds = max_rounds
        self.stalemate_threshold = stalemate_threshold

        # Set up registry with default provider if only provider is given
        if provider and not provider_registry:
            self.provider_registry = ProviderRegistry()
            self.provider_registry.set_default(provider)

    def _get_provider_for_persona(self, persona: Persona) -> LLMProvider:
        """Get the appropriate provider for a persona.

        Resolution order:
        1. Persona's provider_config (if set)
        2. Provider registry lookup by persona name
        3. Default provider
        """
        # If persona has explicit provider_config, create provider from it
        if persona.provider_config:
            cfg = persona.provider_config
            return create_provider(
                model=cfg.model or (self.provider.config.model if self.provider else "openai/qwen/qwen3-coder-30b"),
                api_base=cfg.api_base or (self.provider.config.api_base if self.provider else None),
                api_key=cfg.api_key or (self.provider.config.api_key if self.provider else None),
                temperature=cfg.temperature or 0.7,
                max_tokens=cfg.max_tokens or cfg.response_size or 1024,
            )

        # Try registry lookup
        if self.provider_registry:
            try:
                return self.provider_registry.get_for_persona(persona.name)
            except (ValueError, KeyError):
                pass  # Fall through to default

        # Fall back to single provider
        if self.provider:
            return self.provider

        # Last resort: get default from registry
        if self.provider_registry:
            return self.provider_registry.get_default()

        raise ValueError("No provider available for council engine")

    def run_session(
        self,
        topic: str,
        objective: str,
        personas: list[Persona],
        initial_context: Optional[str] = None,
    ) -> CouncilSession:
        """Run a complete council session.

        Args:
            topic: The topic being discussed
            objective: The goal/decision to reach
            personas: List of personas participating
            initial_context: Optional context to start discussion

        Returns:
            Complete session with results
        """
        session = CouncilSession(
            topic=topic,
            objective=objective,
            personas=personas,
        )

        # Build discussion history
        history: list[Message] = []
        stalemate_counter = 0
        last_consensus_check = None

        for round_num in range(1, self.max_rounds + 1):
            # Conduct discussion round
            round_result = self._conduct_round(
                round_num=round_num,
                topic=topic,
                objective=objective,
                personas=personas,
                history=history,
                initial_context=initial_context if round_num == 1 else None,
            )

            session.rounds.append(round_result)
            history.extend(round_result.messages)

            # Check for consensus
            consensus = self._check_consensus(
                topic=topic,
                objective=objective,
                history=history,
                personas=personas,
            )

            if consensus["reached"]:
                session.consensus_reached = True
                session.final_consensus = consensus["position"]
                round_result.consensus_reached = True
                round_result.consensus_position = consensus["position"]
                break

            # Check for stalemate
            if last_consensus_check == consensus.get("summary"):
                stalemate_counter += 1
            else:
                stalemate_counter = 0
                last_consensus_check = consensus.get("summary")

            if stalemate_counter >= self.stalemate_threshold:
                # Force a vote
                vote_result = self._conduct_vote(
                    topic=topic,
                    objective=objective,
                    personas=personas,
                    history=history,
                )
                round_result.votes = vote_result["votes"]

                if vote_result["consensus_reached"]:
                    session.consensus_reached = True
                    session.final_consensus = vote_result["position"]
                    break

        # If we hit max rounds without consensus, do final vote
        if not session.consensus_reached:
            final_vote = self._conduct_vote(
                topic=topic,
                objective=objective,
                personas=personas,
                history=history,
            )
            if session.rounds:
                session.rounds[-1].votes = final_vote["votes"]
            session.final_consensus = final_vote.get("position", "No consensus reached")
            session.consensus_reached = final_vote.get("consensus_reached", False)

        return session

    def _conduct_round(
        self,
        round_num: int,
        topic: str,
        objective: str,
        personas: list[Persona],
        history: list[Message],
        initial_context: Optional[str] = None,
    ) -> RoundResult:
        """Conduct a single discussion round."""
        messages: list[Message] = []

        # Build history context
        history_text = self._format_history(history)

        for persona in personas:
            user_prompt = self._build_discussion_prompt(
                round_num=round_num,
                topic=topic,
                objective=objective,
                history_text=history_text,
                initial_context=initial_context,
                other_messages=messages,  # Messages from this round
            )

            # Get per-persona provider
            persona_provider = self._get_provider_for_persona(persona)
            response = persona_provider.complete(
                system_prompt=persona.to_system_prompt(),
                user_prompt=user_prompt,
            )

            message = Message(
                persona_name=persona.name,
                content=response.strip(),
                round_number=round_num,
                message_type="discussion",
            )
            messages.append(message)

        return RoundResult(
            round_number=round_num,
            messages=messages,
            consensus_reached=False,
        )

    def _build_discussion_prompt(
        self,
        round_num: int,
        topic: str,
        objective: str,
        history_text: str,
        initial_context: Optional[str],
        other_messages: list[Message],
    ) -> str:
        """Build the prompt for a discussion turn."""
        parts = [f"TOPIC: {topic}", f"OBJECTIVE: {objective}"]

        if initial_context:
            parts.append(f"CONTEXT: {initial_context}")

        if history_text:
            parts.append(f"PREVIOUS DISCUSSION:\n{history_text}")

        if other_messages:
            current_round = "\n".join(
                f"- {m.persona_name}: {m.content}" for m in other_messages
            )
            parts.append(f"THIS ROUND SO FAR:\n{current_round}")

        parts.append(
            f"\nThis is round {round_num}. Please contribute your perspective. "
            "Be constructive and work toward the objective. "
            "If you agree with emerging consensus, say so. "
            "If you disagree, explain why and propose alternatives."
        )

        return "\n\n".join(parts)

    def _format_history(self, history: list[Message]) -> str:
        """Format message history as text."""
        if not history:
            return ""

        rounds: dict[int, list[str]] = {}
        for msg in history:
            if msg.round_number not in rounds:
                rounds[msg.round_number] = []
            rounds[msg.round_number].append(f"  - {msg.persona_name}: {msg.content}")

        parts = []
        for round_num in sorted(rounds.keys()):
            parts.append(f"Round {round_num}:")
            parts.extend(rounds[round_num])

        return "\n".join(parts)

    def _check_consensus(
        self,
        topic: str,
        objective: str,
        history: list[Message],
        personas: list[Persona],
    ) -> dict:
        """Check if consensus has been reached."""
        if not history:
            return {"reached": False}

        history_text = self._format_history(history)

        system_prompt = """You are a neutral moderator analyzing a discussion for consensus.
Determine if the participants have reached agreement on the objective.

Respond with a JSON object:
{
    "reached": true/false,
    "position": "The consensus position if reached, or null",
    "summary": "Brief summary of current state",
    "disagreements": ["List of remaining disagreements if any"]
}

Only output the JSON, nothing else."""

        user_prompt = f"""Topic: {topic}
Objective: {objective}
Participants: {', '.join(p.name for p in personas)}

Discussion:
{history_text}

Has consensus been reached?"""

        try:
            # Use default provider for moderation tasks
            moderator_provider = self.provider or (self.provider_registry.get_default() if self.provider_registry else None)
            if not moderator_provider:
                return {"reached": False, "summary": "No provider available"}

            response = moderator_provider.complete(system_prompt, user_prompt)
            # Parse JSON from response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            return json.loads(response)
        except (json.JSONDecodeError, Exception):
            return {"reached": False, "summary": "Unable to determine consensus"}

    def _conduct_vote(
        self,
        topic: str,
        objective: str,
        personas: list[Persona],
        history: list[Message],
    ) -> dict:
        """Conduct a vote among personas."""
        votes: list[Vote] = []
        history_text = self._format_history(history)

        # Get the current proposal to vote on
        proposal = self._synthesize_proposal(topic, objective, history_text)

        for persona in personas:
            vote = self._get_vote(persona, topic, objective, proposal, history_text)
            votes.append(vote)

        # Tally results
        agree_count = sum(1 for v in votes if v.choice == VoteChoice.AGREE)
        total_voting = sum(1 for v in votes if v.choice != VoteChoice.ABSTAIN)

        if total_voting == 0:
            return {
                "votes": votes,
                "consensus_reached": False,
                "position": None,
            }

        ratio = agree_count / total_voting

        # Check against consensus type
        thresholds = {
            ConsensusType.UNANIMOUS: 1.0,
            ConsensusType.SUPERMAJORITY: 2/3,
            ConsensusType.MAJORITY: 0.5,
            ConsensusType.PLURALITY: 0,  # Most votes wins
        }

        threshold = thresholds[self.consensus_type]
        consensus_reached = ratio > threshold

        return {
            "votes": votes,
            "consensus_reached": consensus_reached,
            "position": proposal if consensus_reached else None,
            "agree_count": agree_count,
            "total_voting": total_voting,
            "ratio": ratio,
        }

    def _synthesize_proposal(
        self,
        topic: str,
        objective: str,
        history_text: str,
    ) -> str:
        """Synthesize a proposal from the discussion to vote on."""
        system_prompt = """You are a neutral moderator. Synthesize the discussion into a single proposal for voting.
The proposal should capture the most supported position.
Output ONLY the proposal text, nothing else."""

        user_prompt = f"""Topic: {topic}
Objective: {objective}

Discussion:
{history_text}

Synthesize a clear proposal for the group to vote on:"""

        # Use default provider for moderation tasks
        moderator_provider = self.provider or (self.provider_registry.get_default() if self.provider_registry else None)
        if not moderator_provider:
            return "No consensus proposal available"

        response = moderator_provider.complete(system_prompt, user_prompt)
        return response.strip()

    def _get_vote(
        self,
        persona: Persona,
        topic: str,
        objective: str,
        proposal: str,
        history_text: str,
    ) -> Vote:
        """Get a vote from a persona."""
        user_prompt = f"""Topic: {topic}
Objective: {objective}

The discussion has led to this proposal for a vote:
PROPOSAL: {proposal}

Based on your perspective and the discussion, cast your vote.
You must respond with EXACTLY one of: AGREE, DISAGREE, or ABSTAIN
Then briefly explain your reasoning (1-2 sentences).

Format:
VOTE: [AGREE/DISAGREE/ABSTAIN]
REASON: [Your reasoning]"""

        # Get per-persona provider
        persona_provider = self._get_provider_for_persona(persona)
        response = persona_provider.complete(
            system_prompt=persona.to_system_prompt(),
            user_prompt=user_prompt,
        )

        # Parse vote
        response_upper = response.upper()
        if "AGREE" in response_upper and "DISAGREE" not in response_upper:
            choice = VoteChoice.AGREE
        elif "DISAGREE" in response_upper:
            choice = VoteChoice.DISAGREE
        else:
            choice = VoteChoice.ABSTAIN

        # Extract reasoning
        reasoning = response
        if "REASON:" in response:
            reasoning = response.split("REASON:", 1)[1].strip()
        elif ":" in response:
            reasoning = response.split(":", 1)[1].strip()

        return Vote(
            persona_name=persona.name,
            choice=choice,
            reasoning=reasoning[:500],  # Limit length
        )
