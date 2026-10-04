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
from typing import Callable, Dict, Optional

from .log_store import _slugify


@dataclass
class Delegation:
    action: str  # "reuse" | "spawn" | "clarify"
    agent_id: str = ""
    label: str = ""  # display name of the chosen agent
    question: str = ""  # clarify only


# (agent_name proposed by the IA, instructions) -> Delegation
DelegationRouter = Callable[[str, str], Delegation]

_router: Optional[DelegationRouter] = None
_names: Dict[str, str] = {}  # manager agent_id -> execution-agent name


def tier3_enabled() -> bool:
    return _router is not None or os.getenv("OPENPOKE_ROUTER", "").strip().lower() == "tier3"


def set_delegation_router(router: Optional[DelegationRouter]) -> None:
    """Install a router (tests, benchmarks) or ``None`` to go back to the env default."""
    global _router
    _router = router
    _names.clear()


def route_delegation(agent_name: str, instructions: str) -> Delegation:
    global _router
    if _router is None:
        _router = _default_router()
    return _router(agent_name, instructions)


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


def _default_router() -> DelegationRouter:
    # ponytail: in-memory registry, cards learn only from instructions (no post-run entity
    # updates, no lifecycle maintenance). Persist the registry and feed run outcomes back
    # through AgentManager.observe() for production.
    from overload.config import get_chat_model
    from overload.routing.base import RoutingRequest
    from overload.routing.tier3 import AgentManager

    manager = AgentManager(model=get_chat_model("router"))
    ids = itertools.count(1)

    def route(agent_name: str, instructions: str) -> Delegation:
        decision = manager.route(RoutingRequest(request_id=f"req_{next(ids)}", now=datetime.now(),
                                                instruction=instructions, name_hint=agent_name))
        return Delegation(decision.action, decision.agent_id or "", decision.agent_name or agent_name,
                          decision.clarify_question or "")

    return route
