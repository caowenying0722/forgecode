'''Classify model failures without mutating conversation or context state.'''

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from forge.runtime.model_client import (
    ModelCallError,
    ModelOutputTruncatedError,
    ModelProtocolError,
)
from forge.runtime.protocol_recovery import (
    build_output_continuation_feedback,
    build_protocol_recovery_feedback,
)
from forge.runtime.state import ModelCallFailed, TokenUsage


class ModelFailureAction(StrEnum):
    CONTINUE_OUTPUT = 'continue_output'
    COMPACT_CONTEXT = 'compact_context'
    RECOVER_PROTOCOL = 'recover_protocol'
    RAISE = 'raise'


@dataclass(frozen=True, slots=True)
class ModelFailureDecision:
    action: ModelFailureAction
    event: ModelCallFailed
    feedback: tuple[dict[str, Any], ...] = ()
    consume_usage: bool = False
    preserve_partial_text: bool = False


class ModelFailureHandler:
    '''Turn one provider/protocol exception into an orchestration decision.'''

    def classify(
        self,
        error: Exception,
        *,
        iteration: int,
        partial_text: str,
        has_tool_calls: bool,
        request_usage: TokenUsage | None,
        output_continuations: int,
        max_output_continuations: int,
        reactive_compaction_attempted: bool,
        protocol_recoveries: int,
        max_protocol_recoveries: int,
        available_tools: tuple[str, ...],
    ) -> ModelFailureDecision:
        if (
            isinstance(error, ModelOutputTruncatedError)
            and not error.tool_names
            and not has_tool_calls
            and partial_text.strip()
        ):
            if (
                output_continuations < max_output_continuations
                and request_usage is not None
            ):
                attempt = output_continuations + 1
                return ModelFailureDecision(
                    action=ModelFailureAction.CONTINUE_OUTPUT,
                    event=failed_event(
                        iteration, error.reason, retryable=True
                    ),
                    feedback=(
                        build_output_continuation_feedback(
                            attempt=attempt,
                            maximum=max_output_continuations,
                        ),
                    ),
                    consume_usage=True,
                    preserve_partial_text=True,
                )
            return ModelFailureDecision(
                action=ModelFailureAction.RAISE,
                event=failed_event(
                    iteration, error.reason, retryable=False
                ),
            )
        if (
            isinstance(error, ModelCallError)
            and error.reason == 'context_overflow'
            and not reactive_compaction_attempted
        ):
            return ModelFailureDecision(
                action=ModelFailureAction.COMPACT_CONTEXT,
                event=failed_event(
                    iteration, error.reason, retryable=True
                ),
            )
        if (
            isinstance(error, ModelProtocolError)
            and protocol_recoveries < max_protocol_recoveries
        ):
            attempt = protocol_recoveries + 1
            return ModelFailureDecision(
                action=ModelFailureAction.RECOVER_PROTOCOL,
                event=failed_event(
                    iteration, error.reason, retryable=True
                ),
                feedback=tuple(
                    build_protocol_recovery_feedback(
                        error,
                        attempt=attempt,
                        maximum=max_protocol_recoveries,
                        available_tools=available_tools,
                    )
                ),
                consume_usage=request_usage is not None,
            )
        reason = (
            error.reason
            if isinstance(error, (ModelCallError, ModelProtocolError))
            else type(error).__name__
        )
        retryable = (
            error.retryable if isinstance(error, ModelCallError) else False
        )
        return ModelFailureDecision(
            action=ModelFailureAction.RAISE,
            event=failed_event(iteration, reason, retryable=retryable),
        )


def failed_event(
    iteration: int,
    reason: str,
    *,
    retryable: bool,
) -> ModelCallFailed:
    return ModelCallFailed(
        iteration=iteration,
        reason=reason,
        retryable=retryable,
    )
