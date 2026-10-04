"""Optional Tier 3 routing: the Agent Manager from openpoke-agent-overload.

With ``OPENPOKE_ROUTER=tier3`` the Interaction Agent no longer sees the roster and no
longer picks agents by exact name. ``send_message_to_agent`` hands its instructions to the
Agent Manager, which reuses the agent that owns the people, threads or reminders they
mention, creates a new agent, or asks the user which person they meant.

Needs the ``openpoke-agent-overload`` package (``pip install -e ..`` from the parent repo).
"""

from __future__ import annotations

import inspect
import itertools
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .log_store import _slugify


@dataclass
class Delegation:
    action: str  # "reuse" | "spawn" | "clarify"
    agent_id: str = ""
    label: str = ""  # display name of the chosen agent
    question: str = ""  # clarify only


# (agent_name proposed by the IA, instructions, follow_up declared by the IA) -> Delegation
DelegationRouter = Callable[[str, str, Optional[bool]], Delegation]
# text of the current turn -> rendered cards of the agents most relevant to it
CandidateProvider = Callable[[str], List[str]]
OutcomeObserver = Callable[[str, str, Any], None]

_router: Optional[DelegationRouter] = None
_candidates: Optional[CandidateProvider] = None
_observer: Optional[OutcomeObserver] = None
_names: Dict[str, str] = {}  # manager agent_id -> execution-agent name


def tier3_enabled() -> bool:
    return _router is not None or os.getenv("OPENPOKE_ROUTER", "").strip().lower() == "tier3"


def set_delegation_router(
    router: Optional[Callable[..., Delegation]],
    candidates: Optional[CandidateProvider] = None,
    observer: Optional[OutcomeObserver] = None,
) -> None:
    """Install a router (tests, benchmarks) or ``None`` to go back to the env default."""
    global _router, _candidates, _observer
    _router = _adapt_router(router) if router is not None else None
    _candidates, _observer = candidates, observer
    _names.clear()


def _adapt_router(router: Callable[..., Delegation]) -> DelegationRouter:
    """Keep the original two-argument test/plugin callback contract working."""
    try:
        inspect.signature(router).bind("agent", "instructions", None)
    except TypeError:
        return lambda name, instructions, follow_up: router(name, instructions)
    except (ValueError, AttributeError):  # callable objects without an inspectable signature
        pass
    return router


def _ensure_router() -> None:
    global _router, _candidates, _observer
    if _router is None:
        _router, _candidates, _observer = _default_router()


def route_delegation(agent_name: str, instructions: str, follow_up: Optional[bool] = None) -> Delegation:
    _ensure_router()
    return _router(agent_name, instructions, follow_up)


def relevant_agents(text: str) -> List[str]:
    """Cards of the few agents most relevant to ``text`` (shown to the IA instead of the roster)."""
    _ensure_router()
    return _candidates(text) if _candidates else []


def record_execution_outcome(delegation: Delegation, instructions: str, result: Any) -> None:
    """Feed compact resource observations back to the default persistent manager."""
    _ensure_router()
    if _observer is not None and delegation.agent_id:
        _observer(delegation.agent_id, instructions, result)


def execution_agent_name(delegation: Delegation) -> str:
    """One execution agent per manager agent.

    Execution-agent logs are keyed by the slug of the name, so two manager agents with
    the same label ("Email to Alice" for two different Alices) would otherwise share memory.
    """
    name = _names.get(delegation.agent_id)
    if name is None:
        label = delegation.label or delegation.agent_id
        taken = {_slugify(n) for n in _names.values()}
        name = label if _slugify(label) not in taken else f"{label} ({delegation.agent_id})"
        _names[delegation.agent_id] = name
    return name


def intent_of(follow_up: Optional[bool]) -> Optional[str]:
    return None if follow_up is None else ("followup" if follow_up else "create")


def _default_router() -> Tuple[DelegationRouter, CandidateProvider, OutcomeObserver]:
    from overload.config import get_chat_model
    from overload.registry import LIVE_STATUSES, AgentRegistry, AgentStatus
    from overload.routing.base import RoutingRequest, RunOutcome, infer_domain, infer_intent
    from overload.routing.tier3 import AgentManager
    from server.services.conversation import get_conversation_log
    from .roster import get_agent_roster

    state_path = Path(os.getenv(
        "OPENPOKE_AGENT_MANAGER_STATE",
        Path(__file__).resolve().parents[2] / "data" / "execution_agents" / "agent_manager_registry.json",
    )).expanduser()
    registry = AgentRegistry.load(state_path)
    if not registry.all_cards():
        roster = get_agent_roster()
        roster.load()
        now = datetime.now()
        for name in roster.get_agents():
            registry.create(name, now, purpose=name, domain=infer_domain(name))
        registry.save(state_path)

    manager = AgentManager(registry=registry, model=get_chat_model("router"))
    for card in sorted(
        (card for card in registry.all_cards() if card.status != AgentStatus.MERGED),
        key=lambda item: item.agent_id,
    ):
        execution_agent_name(Delegation("reuse", card.agent_id, card.name))
    ids = itertools.count(1)

    def route(agent_name: str, instructions: str, follow_up: Optional[bool] = None) -> Delegation:
        now = datetime.now()
        manager.maintain(now)
        intent = intent_of(follow_up)
        if intent == "create" and infer_intent(instructions) == "followup":
            intent = "followup"
        entries = [entry for entry in get_conversation_log().iter_entries()
                   if entry[0] in {"user_message", "poke_reply", "agent_message"}][-12:]
        conversation = [f"{'user' if tag == 'user_message' else 'assistant'}: {text}"
                        for tag, _, text in entries]
        user_message = next((text for tag, _, text in reversed(entries) if tag == "user_message"), "")
        decision = manager.route(RoutingRequest(request_id=f"req_{next(ids)}", now=now,
                                                instruction=instructions, name_hint=agent_name,
                                                intent=intent, user_message=user_message,
                                                conversation=conversation))
        manager.registry.save(state_path)
        return Delegation(decision.action, decision.agent_id or "", decision.agent_name or agent_name,
                          decision.clarify_question or "")

    def candidates(text: str) -> List[str]:
        now = datetime.now()
        return [c.card.render_card(now) for c in manager.index.search(text, now, k=5,
                                                                       statuses=LIVE_STATUSES)]

    def observe(agent_id: str, instructions: str, result: Any) -> None:
        tools = getattr(result, "tools_executed", []) or []
        summary = ""
        if tools:
            status = "Completed" if getattr(result, "success", False) else "Partially executed"
            summary = (status + " with: " + ", ".join(tools))[:400]
        outcome = RunOutcome(
            owned_entities=list(getattr(result, "observed_entities", []) or []),
            open_items_add=dict(getattr(result, "open_items_add", {}) or {}),
            open_items_close=list(getattr(result, "open_items_close", []) or []),
            summary=summary,
        )
        request = RoutingRequest(request_id=f"observe_{next(ids)}", now=datetime.now(), instruction=instructions)
        manager.observe(agent_id, request, outcome)
        manager.registry.save(state_path)

    return route, candidates, observe
