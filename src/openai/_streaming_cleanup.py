from __future__ import annotations

from typing import Optional, AsyncIterator, cast

import anyio
import httpx2
from anyio.lowlevel import checkpoint

from ._httpx2 import request_exceptions, stream_consumed_exceptions

_DRAIN_TIMEOUT = 0.1


async def drain_response_async(response: httpx2.Response) -> None:
    if response.is_closed:
        return
    try:
        with anyio.move_on_after(_DRAIN_TIMEOUT):
            # Discard raw transport bytes without decoding trailing SSE or content.
            async for _ in cast(AsyncIterator[bytes], response.stream):
                # Also enforce the deadline on transports yielding buffered data.
                await checkpoint()
    except request_exceptions() + stream_consumed_exceptions():
        pass  # Cleanup must not turn completed output into a request error.
    finally:
        # Close outside our timeout scope and shield cleanup from caller
        # cancellation. Cancellation still propagates after the response closes.
        with anyio.CancelScope(shield=True):
            await response.aclose()


def close_owned(*iterators: object, primary: Optional[BaseException] = None) -> None:
    """Close iterators this caller opened, in order.

    Every iterator gets its close attempt. Objects without ``close()`` are skipped.
    A cleanup error is raised only when no ``primary`` error is already propagating.
    """
    failure: Optional[BaseException] = None
    for iterator in iterators:
        close = getattr(iterator, "close", None)
        if close is None:
            continue
        try:
            close()
        except Exception as err:
            failure = failure or err
    if failure is not None and primary is None:
        raise failure


async def aclose_owned(*iterators: object, primary: Optional[BaseException] = None) -> None:
    """Async counterpart of ``close_owned``.

    Cleanup is shielded from caller cancellation so an already-cancelled scope cannot
    leave a generator closed with its cleanup half done; cancellation still propagates.
    """
    failure: Optional[BaseException] = None
    with anyio.CancelScope(shield=True):
        for iterator in iterators:
            aclose = getattr(iterator, "aclose", None)
            if aclose is None:
                continue
            try:
                await aclose()
            except Exception as err:
                failure = failure or err
    if failure is not None and primary is None:
        raise failure
