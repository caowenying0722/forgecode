'''Stable browser-smoke tests using fake server and browser adapters.'''

import asyncio

from forge.runtime.browser_smoke import (
    BrowserObservation,
    BrowserUnavailableError,
    BrowserSmokeVerifier,
)


class FakeServer:
    def __init__(self, *, ready: bool = True) -> None:
        self.ready = ready
        self.started_port: int | None = None
        self.stopped = False

    async def start(self, port: int) -> None:
        self.started_port = port

    async def wait_ready(self, port: int, timeout_seconds: float) -> bool:
        del port, timeout_seconds
        return self.ready

    async def stop(self) -> None:
        self.stopped = True


class FakeBrowser:
    def __init__(self, observation: BrowserObservation) -> None:
        self.observation = observation

    async def inspect(self, url: str) -> BrowserObservation:
        del url
        return self.observation


def test_browser_console_error_fails_smoke_verification() -> None:
    server = FakeServer()
    browser = FakeBrowser(
        BrowserObservation(
            http_status=200,
            canvas_count=1,
            console_errors=('Uncaught TypeError',),
        )
    )

    result = asyncio.run(BrowserSmokeVerifier().verify(server, browser))

    assert result.status == 'failed'
    assert result.console_errors == ('Uncaught TypeError',)
    assert result.server_terminated is True


def test_dev_server_is_terminated_after_smoke_test() -> None:
    server = FakeServer()
    browser = FakeBrowser(
        BrowserObservation(http_status=200, canvas_count=1)
    )

    result = asyncio.run(BrowserSmokeVerifier().verify(server, browser))

    assert result.status == 'passed'
    assert server.stopped is True
    assert result.server_terminated is True


def test_http_success_without_canvas_fails_browser_smoke() -> None:
    server = FakeServer()
    browser = FakeBrowser(
        BrowserObservation(http_status=200, canvas_count=0)
    )

    result = asyncio.run(BrowserSmokeVerifier().verify(server, browser))

    assert result.status == 'failed'
    assert any('canvas' in item for item in result.diagnostics)


def test_browser_dependency_absence_is_unavailable_and_stops_server() -> None:
    class UnavailableBrowser:
        async def inspect(self, url: str) -> BrowserObservation:
            del url
            raise BrowserUnavailableError('No browser adapter is installed.')

    server = FakeServer()

    result = asyncio.run(
        BrowserSmokeVerifier().verify(server, UnavailableBrowser())
    )

    assert result.status == 'unavailable'
    assert server.stopped is True
    assert result.server_terminated is True
