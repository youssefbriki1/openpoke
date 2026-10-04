"""Optional Tier 3 routing: the Agent Manager from openpoke-agent-overload.

With ``OPENPOKE_ROUTER=tier3`` the Interaction Agent no longer sees the roster and no
longer picks agents by exact name. ``send_message_to_agent`` hands its instructions to the
Agent Manager, which reuses the agent that owns the people, threads or reminders they
mention, creates a new agent, or asks the user which person they meant.

Needs the ``openpoke-agent-overload`` package (``pip install -e ..`` from the parent repo).
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

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

_router: Optional[DelegationRouter] = None
_candidates: Optional[CandidateProvider] = None
_names: Dict[str, str] = {}  # manager agent_id -> execution-agent name


def tier3_enabled() -> bool:
    return _router is not None or os.getenv("OPENPOKE_ROUTER", "").strip().lower() == "tier3"


def set_delegation_router(router: Optional[DelegationRouter], candidates: Optional[CandidateProvider] = None) -> None:
    """Install a router (tests, benchmarks) or ``None`` to go back to the env default."""
    global _router, _candidates
    _router, _candidates = router, candidates
    _names.clear()


def _ensure_router() -> None:
    global _router, _candidates
    if _router is None:
        _router, _candidates = _default_router()


def route_delegation(agent_name: str, instructions: str, follow_up: Optional[bool] = None) -> Delegation:
    _ensure_router()
    return _router(agent_name, instructions, follow_up)


def relevant_agents(text: str) -> List[str]:
    """Cards of the few agents most relevant to ``text`` (shown to the IA instead of the roster)."""
    _ensure_router()
    return _candidates(text) if _candidates else []


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


def _default_router() -> Tuple[DelegationRouter, CandidateProvider]:
    # ponytail: in-memory registry, cards learn only from instructions (no post-run entity
    # updates, no lifecycle maintenance). Persist the registry and feed run outcomes back
    # through AgentManager.observe() for production.
    from overload.config import get_chat_model
    from overload.routing.base import RoutingRequest
    from overload.routing.tier3 import AgentManager

    manager = AgentManager(model=get_chat_model("router"))
    ids = itertools.count(1)

    def route(agent_name: str, instructions: str, follow_up: Optional[bool]) -> Delegation:
        decision = manager.route(RoutingRequest(request_id=f"req_{next(ids)}", now=datetime.now(),
                                                instruction=instructions, name_hint=agent_name,
                                                intent=intent_of(follow_up)))
        return Delegation(decision.action, decision.agent_id or "", decision.agent_name or agent_name,
                          decision.clarify_question or "")

    def candidates(text: str) -> List[str]:
        now = datetime.now()
        return [c.card.render_card(now) for c in manager.index.search(text, now, k=5)]

    return route, candidates
