"""Shared constructor argument normalization; no execution or frontier algorithms."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from xgraph.tokenizer import ServedTokenizer


@dataclass(frozen=True)
class HostPolicy:
    end_after_answer: bool
    eos_token_ids: frozenset[int] | tuple[int, ...]
    observe_select: bool
    observe_answer: bool


def normalize_host_policy(
    tokenizer: ServedTokenizer,
    *,
    paths: bool,
    relations: str,
    expressions: bool,
    stop: str | None,
    end_after_answer: bool,
    eos_token_ids: Iterable[int],
    observe: int,
    observe_width: int,
    observe_defer: int,
    menu_close_next: bool,
    observe_select: bool,
    observe_answer: bool,
    observe_required: bool,
    observe_no_loop: bool,
    audit: bool,
    audit_entities: int,
    audit_relations: int,
    menu: int,
    observe_terminal: int,
    terminal_order: str,
    terminal_seed: int,
    request_path_options: bool = False,
) -> HostPolicy:
    """Check display policies in constructor order, retaining the capability union.

    Call after checking menu arguments and any independent menu tokenization.
    Explicit end-after-answer/path checks precede this function; the implied
    setting retains its existing EOS-first validation order.
    """
    observe_answer = observe_answer or menu_close_next or bool(observe_defer)
    end_after_answer = end_after_answer or observe_answer or audit or bool(observe_terminal)
    eos = tokenizer.check_eos_token_ids(eos_token_ids) if end_after_answer else ()
    from xgraph.relations import check_relations

    check_relations(relations, paths=paths, expressions=expressions, stop=stop)
    from xgraph.observations import check_observe, check_observed_options

    check_observe(observe, paths=paths, relations=relations, expressions=expressions)
    check_observed_options(
        observe,
        observe_select,
        observe_answer,
        observe_required,
        observe_no_loop,
        width=observe_width,
        defer=observe_defer,
        close_next=menu_close_next,
        menu=menu,
    )
    from xgraph.audit import check_audit, check_terminal, check_terminal_order

    observed = (observe, observe_select, observe_answer, observe_required, observe_no_loop)
    check_terminal(
        observe_terminal,
        paths=paths,
        relations=relations,
        expressions=expressions,
        incompatible=() if request_path_options else (audit, menu, *observed),
    )
    check_terminal_order(terminal_order, terminal_seed, observe_terminal)
    check_audit(
        audit,
        audit_entities,
        audit_relations,
        paths=paths,
        relations=relations,
        expressions=expressions,
        observed=observed,
    )
    return HostPolicy(end_after_answer, eos, observe_select or observe_required, observe_answer)
