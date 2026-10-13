DEFAULT_MAX_DRAFT_REVISIONS = 3


class MaxDraftRevisionsExceededError(RuntimeError):
    """Raised when reviewer/reviser rounds exceed the configured draft limit."""


def route_draft_review(draft, max_draft_revisions=DEFAULT_MAX_DRAFT_REVISIONS):
    """针对“接受/修订”这一编辑评审循环的逻辑处理：

*   若 `review` 为 `None` → 接受（包括在达到硬性上限触发强制接受、且调用方已清除 `review` 状态的情况）。
*   若 `max_draft_revisions` 为 `None` → 永不强制接受（即由操作员选择不启用此限制）。
*   若 `draft_revision_count > max` → 抛出异常，以便边缘逻辑（edge）能优雅地执行强制接受，从而避免触发 LangGraph 的递归深度限制（recursion_limit）。
    """
    if draft.get("review") is None:
        return "accept"

    if max_draft_revisions is None:
        return "revise"

    count = draft.get("draft_revision_count", 0)
    if count > max_draft_revisions:
        raise MaxDraftRevisionsExceededError(
            "Draft revision limit exceeded. "
            f"Received {count} revision rounds; "
            f"max_draft_revisions is {max_draft_revisions}."
        )

    return "revise"
