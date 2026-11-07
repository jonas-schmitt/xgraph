"""The masked-policy loss: cross-entropy over the tokens the constraint allows.

Constrained decoding samples from the model's distribution renormalized over the
constraint's allowed set. Training on that policy (method pilot 1 in
docs/research-plan.md) needs, at every reply position, the allowed set of the state
the reply has reached under the example's KG. The engine computes all of them in one
call: the reply is masked as one request's block of draft rows (Engine.apply_mask with
local_pos and input_ids), each row as if the reply tokens before it were committed.
Outside spans the constraint allows every token, so the loss there is the ordinary
cross-entropy. For a path-writing reply the engine runs in path mode (xgraph.train
--paths), and an anchored request's reply starts from its anchor (Engine.anchor).
"""

from __future__ import annotations

SLOT = 0


def mask_reply(engine, logits, reply: list[int], anchor: list[str] | None = None):
    """Set the disallowed logits of each reply position to -inf, in place, and return them.

    logits is [len(reply), columns] float32 on the GPU; row i predicts reply[i] from
    the state after reply[:i], the constraint starting in free text, bound to the
    anchor's entities (QIDs) when one is given.
    """
    import torch

    rows = len(reply)
    device = logits.device
    slots = torch.tensor([SLOT], dtype=torch.int32, device=device)
    engine.reset()
    engine.begin(slots)
    if anchor:
        engine.anchor(SLOT, anchor)
    engine.apply_mask(
        logits,
        torch.zeros(rows, dtype=torch.int32, device=device),
        local_pos=torch.arange(rows, dtype=torch.int32, device=device),
        input_ids=torch.tensor([0, *reply[:-1]], dtype=torch.int32, device=device),
    )
    return logits


def masked_loss(
    engine,
    logits,
    reply: list[int],
    anchor: list[str] | None = None,
    keep: list[bool] | None = None,
):
    """Mean cross-entropy of the reply under the constraint's renormalized distribution,
    over the positions keep marks (every position by default); a left-out position still
    moves the constraint's state but carries no loss, and is not checked."""
    import torch

    masked = mask_reply(engine, logits.float().clone(), reply, anchor)
    targets = torch.tensor(reply, device=logits.device)
    if keep is not None:
        rows = torch.tensor([i for i, kept in enumerate(keep) if kept], device=logits.device)
        masked, targets = masked[rows], targets[rows]
    loss = torch.nn.functional.cross_entropy(masked, targets)
    if not torch.isfinite(loss):
        raise ValueError("the reply leaves the constraint: a target token is disallowed")
    return loss
