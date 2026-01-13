"""Core council discussion engine with isolated persona sessions."""

import json
import logging
from typing import Optional

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
from .voting import VoteParser, VotingMachine, StructuredVote, VOTE_PROMPT_TEMPLATE
from .discussion import (
    ResponseParser,
    ResponseType,
    DiscussionState,
    DiscussionPhase,
    PASS_INSTRUCTION,
)
from .mediator import MediatorRole, select_mediator, reorder_personas_mediator_first

logger = logging.getLogger(__name__)


class CouncilEngine:
    """Engine for running council discussions with isolated persona sessions.

    Key features:
    - Each persona runs as a separate LLM invocation with its own system prompt
    - Deterministic vote parsing and tallying via VotingMachine
    - Mediator persona controls discussion flow
    - PASS mechanism allows personas to defer
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        provider_registry: Optional[ProviderRegistry] = None,
        consensus_type: ConsensusType = ConsensusType.MAJORITY,
        max_rounds: int = 5,
        stalemate_threshold: int = 2,
        mediator_index: int = 0,
        allow_pass: bool = True,
        strict_voting: bool = True,
    ):
        """Initialize the council engine.

        Args:
            provider: Single LLM provider for all responses (legacy mode)
            provider_registry: Registry for per-persona provider resolution
            consensus_type: Type of consensus required
            max_rounds: Maximum discussion rounds before forcing vote
            stalemate_threshold: Rounds without progress before calling stalemate
            mediator_index: Index of persona to act as mediator (default: 0)
            allow_pass: Allow personas to pass/defer (default: True)
            strict_voting: Use deterministic VotingMachine (default: True)

        Note: Either provider or provider_registry must be provided.
        """
        self.provider = provider
        self.provider_registry = provider_registry
        self.consensus_type = consensus_type
        self.max_rounds = max_rounds
        self.stalemate_threshold = stalemate_threshold
        self.mediator_index = mediator_index
        self.allow_pass = allow_pass
        self.strict_voting = strict_voting

        # Initialize voting machine
        self.voting_machine = VotingMachine(consensus_type)

        # Set up registry with default provider if only provider is given
        if provider and not provider_registry:
            self.provider_registry = ProviderRegistry()
            self.provider_registry.set_default(provider)

    def _get_provider_for_persona(self, persona: Persona) -> LLMProvider:
        """Get the appropriate provider for a persona.

        Each persona gets its own isolated LLM invocation.

        Resolution order:
        1. Persona's provider_config (if set)
        2. Provider registry lookup by persona name
        3. Default provider
        """
        # If persona has explicit provider_config, create provider from it
        if persona.provider_config:
            cfg = persona.provider_config
            # Get fallback values from default provider if available
            default_config = self.provider.config if self.provider else None
            provider = create_provider(
                model=cfg.model or (default_config.model if default_config else "openai/gpt-4o-mini"),
                api_base=cfg.api_base or (default_config.api_base if default_config else None),
                api_key=cfg.api_key or (default_config.api_key if default_config else None),
                temperature=cfg.temperature if cfg.temperature is not None else (default_config.temperature if default_config else 0.7),
                top_p=cfg.top_p if cfg.top_p is not None else (default_config.top_p if default_config else None),
                top_k=cfg.top_k if cfg.top_k is not None else (default_config.top_k if default_config else None),
                max_tokens=cfg.max_tokens or (default_config.max_tokens if default_config else 1024),
                frequency_penalty=cfg.frequency_penalty if cfg.frequency_penalty is not None else (default_config.frequency_penalty if default_config else None),
                presence_penalty=cfg.presence_penalty if cfg.presence_penalty is not None else (default_config.presence_penalty if default_config else None),
                repeat_penalty=cfg.repeat_penalty if cfg.repeat_penalty is not None else (default_config.repeat_penalty if default_config else None),
                stop=cfg.stop if cfg.stop is not None else (default_config.stop if default_config else None),
                seed=cfg.seed if cfg.seed is not None else (default_config.seed if default_config else None),
                timeout=cfg.timeout or (default_config.timeout if default_config else 120),
            )
            logger.debug(f"Created isolated provider for persona '{persona.name}' from config")
            return provider

        # Try registry lookup
        if self.provider_registry:
            try:
                provider = self.provider_registry.get_for_persona(persona.name)
                logger.debug(f"Got provider for persona '{persona.name}' from registry")
                return provider
            except (ValueError, KeyError):
                pass  # Fall through to default

        # Fall back to single provider
        if self.provider:
            logger.debug(f"Using default provider for persona '{persona.name}'")
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
        """Run a complete council session with isolated persona sessions.

        Each persona participates via separate LLM invocations with their
        unique system prompts. The mediator controls flow and synthesis.

        Args:
            topic: The topic being discussed
            objective: The goal/decision to reach
            personas: List of personas participating
            initial_context: Optional context to start discussion

        Returns:
            Complete session with results
        """
        # Set up mediator
        mediator, mediator_idx = select_mediator(personas, self.mediator_index)
        ordered_personas = reorder_personas_mediator_first(personas, mediator_idx)

        # Mark mediator in persona list
        for i, p in enumerate(ordered_personas):
            if i == 0:  # Mediator is first after reordering
                ordered_personas[i] = Persona(
                    name=p.name,
                    role=p.role,
                    expertise=p.expertise,
                    personality_traits=p.personality_traits,
                    perspective=p.perspective,
                    provider_config=p.provider_config,
                    is_mediator=True,
                )

        session = CouncilSession(
            topic=topic,
            objective=objective,
            personas=ordered_personas,
        )

        # Initialize discussion state
        discussion_state = DiscussionState()

        # Build discussion history
        history: list[Message] = []
        stalemate_counter = 0
        last_positions: set[str] = set()

        logger.info(f"Starting council session: {topic}")
        logger.info(f"Mediator: {ordered_personas[0].name}")
        logger.info(f"Personas: {[p.name for p in ordered_personas]}")

        for round_num in range(1, self.max_rounds + 1):
            discussion_state.advance_round()
            logger.info(f"=== Round {round_num} ({discussion_state.phase.value}) ===")

            # Conduct discussion round with isolated persona sessions
            round_result = self._conduct_round(
                round_num=round_num,
                topic=topic,
                objective=objective,
                personas=ordered_personas,
                history=history,
                initial_context=initial_context if round_num == 1 else None,
                discussion_state=discussion_state,
            )

            session.rounds.append(round_result)
            history.extend(round_result.messages)

            # Check if mediator called for vote
            if discussion_state.vote_called:
                logger.info("Mediator called for vote")
                vote_result = self._conduct_vote(
                    topic=topic,
                    objective=objective,
                    personas=ordered_personas,
                    history=history,
                    proposal=discussion_state.current_proposal,
                )
                round_result.votes = vote_result["votes"]

                if vote_result["consensus_reached"]:
                    session.consensus_reached = True
                    session.final_consensus = vote_result["position"]
                    round_result.consensus_reached = True
                    round_result.consensus_position = vote_result["position"]
                    logger.info(f"Consensus reached via vote: {vote_result['position'][:100]}...")
                    break
                else:
                    # Vote failed, reset and continue
                    discussion_state.vote_called = False
                    discussion_state.phase = DiscussionPhase.DELIBERATION

            # Check for stalemate via position tracking
            current_positions = {m.content[:100] for m in round_result.messages if not m.is_pass}
            if current_positions == last_positions:
                stalemate_counter += 1
            else:
                stalemate_counter = 0
                last_positions = current_positions

            # Auto-vote on stalemate or high pass rate
            if stalemate_counter >= self.stalemate_threshold or discussion_state.should_auto_vote(len(ordered_personas)):
                logger.info(f"Auto-triggering vote (stalemate={stalemate_counter}, passes={discussion_state.total_passes})")
                vote_result = self._conduct_vote(
                    topic=topic,
                    objective=objective,
                    personas=ordered_personas,
                    history=history,
                )
                round_result.votes = vote_result["votes"]

                if vote_result["consensus_reached"]:
                    session.consensus_reached = True
                    session.final_consensus = vote_result["position"]
                    round_result.consensus_reached = True
                    round_result.consensus_position = vote_result["position"]
                    break

        # Final vote if no consensus yet
        if not session.consensus_reached:
            logger.info("Max rounds reached, conducting final vote")
            final_vote = self._conduct_vote(
                topic=topic,
                objective=objective,
                personas=ordered_personas,
                history=history,
            )
            if session.rounds:
                session.rounds[-1].votes = final_vote["votes"]
            session.final_consensus = final_vote.get("position", "No consensus reached")
            session.consensus_reached = final_vote.get("consensus_reached", False)

        logger.info(f"Session complete. Consensus: {session.consensus_reached}")
        return session

    def _conduct_round(
        self,
        round_num: int,
        topic: str,
        objective: str,
        personas: list[Persona],
        history: list[Message],
        initial_context: Optional[str],
        discussion_state: DiscussionState,
    ) -> RoundResult:
        """Conduct a single discussion round with isolated persona sessions.

        Each persona gets its own LLM invocation with persona-specific system prompt.
        """
        messages: list[Message] = []
        history_text = self._format_history(history)

        for i, persona in enumerate(personas):
            is_mediator = (i == 0)  # First persona is always mediator

            # Build persona-specific prompt
            if is_mediator:
                mediator_role = MediatorRole(persona, i)
                user_prompt = mediator_role.get_discussion_prompt(
                    round_num=round_num,
                    topic=topic,
                    objective=objective,
                    history_text=history_text,
                    state=discussion_state,
                )
                system_prompt = mediator_role.get_system_prompt()
            else:
                user_prompt = self._build_discussion_prompt(
                    round_num=round_num,
                    topic=topic,
                    objective=objective,
                    history_text=history_text,
                    initial_context=initial_context,
                    other_messages=messages,
                )
                system_prompt = persona.to_system_prompt()

            # ISOLATED LLM INVOCATION for this persona
            logger.info(f"[API CALL] Persona '{persona.name}' (mediator={is_mediator})")
            persona_provider = self._get_provider_for_persona(persona)
            response = persona_provider.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            logger.debug(f"[RESPONSE] {persona.name}: {response[:100]}...")

            # Parse response for PASS and other directives
            parsed = ResponseParser.parse(
                persona_name=persona.name,
                response=response,
                is_mediator=is_mediator,
            )
            discussion_state.record_response(parsed)

            # Create message
            message = Message(
                persona_name=persona.name,
                content=response.strip(),
                round_number=round_num,
                message_type="pass" if parsed.response_type == ResponseType.PASS else "discussion",
                is_pass=(parsed.response_type == ResponseType.PASS),
                is_mediator=is_mediator,
            )
            messages.append(message)

            logger.info(f"  {persona.name}: {parsed.response_type.value} ({'PASS' if message.is_pass else response[:50] + '...'})")

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

        if self.allow_pass:
            parts.append(PASS_INSTRUCTION)

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
            prefix = "[MEDIATOR] " if msg.is_mediator else ""
            suffix = " [PASS]" if msg.is_pass else ""
            rounds[msg.round_number].append(f"  - {prefix}{msg.persona_name}{suffix}: {msg.content}")

        parts = []
        for round_num in sorted(rounds.keys()):
            parts.append(f"Round {round_num}:")
            parts.extend(rounds[round_num])

        return "\n".join(parts)

    def _conduct_vote(
        self,
        topic: str,
        objective: str,
        personas: list[Persona],
        history: list[Message],
        proposal: Optional[str] = None,
    ) -> dict:
        """Conduct a vote with DETERMINISTIC tallying.

        Uses VotingMachine for deterministic vote counting.
        """
        history_text = self._format_history(history)

        # Get proposal - from mediator or synthesize
        if not proposal:
            proposal = self._synthesize_proposal(topic, objective, history_text)

        logger.info(f"Voting on proposal: {proposal[:100]}...")

        # Collect votes via isolated LLM calls
        structured_votes: list[StructuredVote] = []
        legacy_votes: list[Vote] = []

        for persona in personas:
            if persona.is_mediator:
                # Mediator doesn't vote
                continue

            # Build vote prompt
            vote_prompt = VOTE_PROMPT_TEMPLATE.format(proposal=proposal)
            full_prompt = f"""Topic: {topic}
Objective: {objective}

Discussion summary:
{history_text[-2000:] if len(history_text) > 2000 else history_text}

{vote_prompt}"""

            # ISOLATED LLM INVOCATION for vote
            logger.info(f"[VOTE API CALL] Persona '{persona.name}'")
            persona_provider = self._get_provider_for_persona(persona)
            response = persona_provider.complete(
                system_prompt=persona.to_system_prompt(),
                user_prompt=full_prompt,
            )

            # DETERMINISTIC vote parsing
            structured = VoteParser.parse(persona.name, response)
            structured_votes.append(structured)
            legacy_votes.append(VoteParser.to_legacy_vote(structured))

            logger.info(f"  {persona.name}: {structured.choice.value} (confidence: {structured.confidence:.2f})")
            if structured.parse_errors:
                logger.warning(f"    Parse errors: {structured.parse_errors}")

        # DETERMINISTIC tallying
        tally = self.voting_machine.tally(structured_votes)

        logger.info(f"Vote tally: {tally.agree_count} agree, {tally.disagree_count} disagree, {tally.abstain_count} abstain")
        logger.info(f"Ratio: {tally.agree_ratio:.2%}, Consensus: {tally.consensus_reached}")

        return {
            "votes": legacy_votes,
            "structured_votes": structured_votes,
            "tally": self.voting_machine.to_dict(tally),
            "consensus_reached": tally.consensus_reached,
            "position": proposal if tally.consensus_reached else None,
            "agree_count": tally.agree_count,
            "total_voting": tally.total_voting,
            "ratio": tally.agree_ratio,
        }

    def _synthesize_proposal(
        self,
        topic: str,
        objective: str,
        history_text: str,
    ) -> str:
        """Synthesize a proposal from the discussion to vote on.

        Note: This still uses LLM for synthesis, but voting is deterministic.
        """
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
