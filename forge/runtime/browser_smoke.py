'''Optional, evidence-producing browser smoke verification adapters.'''

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
import socket
from time import perf_counter
from typing import Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from forge.runtime.process import terminate_process_tree
from forge.runtime.state import VerificationEvidence


class BrowserUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BrowserObservation:
    http_status: int
    canvas_count: int
    console_errors: tuple[str, ...] = ()
    page_errors: tuple[str, ...] = ()
    crashed: bool = False


@dataclass(frozen=True, slots=True)
class BrowserSmokeResult:
    status: Literal['passed', 'failed', 'unavailable']
    url: str
    http_status: int | None = None
    canvas_count: int = 0
    console_errors: tuple[str, ...] = ()
    page_errors: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    server_terminated: bool = False

    @property
    def success(self) -> bool:
        return self.status == 'passed'

    def to_evidence(
        self,
        *,
        source_revision: int,
        duration_seconds: float,
    ) -> VerificationEvidence:
        levels = (
            ('dev_server_verified', 'browser_smoke_verified')
            if self.success
            else ()
        )
        return VerificationEvidence(
            command=f'browser smoke {self.url}',
            cwd='.',
            exit_code=0 if self.success else -1,
            duration_seconds=duration_seconds,
            timed_out=False,
            workspace_revision=source_revision,
            source_revision=source_revision,
            status=self.status,
            verification_type='browser_smoke',
            verification_levels=levels,
        )


class DevServerAdapter(Protocol):
    async def start(self, port: int) -> None: ...

    async def wait_ready(self, port: int, timeout_seconds: float) -> bool: ...

    async def stop(self) -> None: ...


class BrowserAdapter(Protocol):
    async def inspect(self, url: str) -> BrowserObservation: ...


class BrowserSmokeVerifier:
    def __init__(
        self,
        *,
        host: str = '127.0.0.1',
        port: int | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_seconds = timeout_seconds

    async def verify(
        self,
        server: DevServerAdapter,
        browser: BrowserAdapter,
    ) -> BrowserSmokeResult:
        port = self.port or available_tcp_port(self.host)
        url = f'http://{self.host}:{port}'
        diagnostics: list[str] = []
        observation: BrowserObservation | None = None
        status: Literal['passed', 'failed', 'unavailable'] = 'failed'
        terminated = False
        try:
            await server.start(port)
            ready = await asyncio.wait_for(
                server.wait_ready(port, self.timeout_seconds),
                timeout=self.timeout_seconds + 1,
            )
            if not ready:
                diagnostics.append('The development server did not become ready.')
            else:
                observation = await asyncio.wait_for(
                    browser.inspect(url),
                    timeout=self.timeout_seconds,
                )
                if observation.http_status < 200 or observation.http_status >= 400:
                    diagnostics.append(
                        f'The page returned HTTP {observation.http_status}.'
                    )
                if observation.canvas_count < 1:
                    diagnostics.append('No canvas element was observed.')
                if observation.console_errors:
                    diagnostics.append('The browser emitted console errors.')
                if observation.page_errors:
                    diagnostics.append('The page emitted uncaught errors.')
                if observation.crashed:
                    diagnostics.append('The page crashed during observation.')
                status = 'passed' if not diagnostics else 'failed'
        except BrowserUnavailableError as error:
            status = 'unavailable'
            diagnostics.append(str(error))
        except Exception as error:
            status = 'failed'
            diagnostics.append(f'{type(error).__name__}: {error}')
        finally:
            try:
                await asyncio.wait_for(server.stop(), timeout=5)
                terminated = True
            except Exception as error:
                diagnostics.append(
                    f'Development server termination failed: {error}'
                )
                status = 'failed'
        return BrowserSmokeResult(
            status=status,
            url=url,
            http_status=(observation.http_status if observation else None),
            canvas_count=(observation.canvas_count if observation else 0),
            console_errors=(observation.console_errors if observation else ()),
            page_errors=(observation.page_errors if observation else ()),
            diagnostics=tuple(diagnostics),
            server_terminated=terminated,
        )


class SubprocessDevServer:
    '''Start an optional project server command with a configurable port.'''

    def __init__(self, command: str, cwd: Path, *, host: str = '127.0.0.1') -> None:
        self.command = command
        self.cwd = cwd.resolve()
        self.host = host
        self.process: asyncio.subprocess.Process | None = None

    async def start(self, port: int) -> None:
        rendered = self.command.replace('{port}', str(port))
        process_env = {**os.environ, 'PORT': str(port)}
        process_group = (
            {'creationflags': 0x00000200}
            if os.name == 'nt'
            else {'start_new_session': True}
        )
        self.process = await asyncio.create_subprocess_shell(
            rendered,
            cwd=self.cwd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env=process_env,
            **process_group,
        )

    async def wait_ready(self, port: int, timeout_seconds: float) -> bool:
        deadline = perf_counter() + timeout_seconds
        url = f'http://{self.host}:{port}'
        while perf_counter() < deadline:
            if self.process is None or self.process.returncode is not None:
                return False
            status = await asyncio.to_thread(_http_status, url)
            if status is not None and 200 <= status < 500:
                return True
            await asyncio.sleep(0.1)
        return False

    async def stop(self) -> None:
        if self.process is not None:
            await terminate_process_tree(self.process)
            self.process = None


class PlaywrightBrowser:
    '''Use Playwright when installed; otherwise report explicit unavailability.'''

    async def inspect(self, url: str) -> BrowserObservation:
        try:
            from playwright.async_api import async_playwright
        except ImportError as error:
            raise BrowserUnavailableError(
                'Browser smoke verification is unavailable because Playwright '
                'is not installed.'
            ) from error
        console_errors: list[str] = []
        page_errors: list[str] = []
        crashed = False
        async with async_playwright() as playwright:
            try:
                browser = await playwright.chromium.launch(headless=True)
            except Exception as error:
                raise BrowserUnavailableError(
                    f'Browser smoke verification is unavailable: {error}'
                ) from error
            try:
                page = await browser.new_page()
                page.on(
                    'console',
                    lambda message: console_errors.append(message.text)
                    if message.type == 'error'
                    else None,
                )
                page.on(
                    'pageerror',
                    lambda error: page_errors.append(str(error)),
                )

                def mark_crashed(*_args: object) -> None:
                    nonlocal crashed
                    crashed = True

                page.on('crash', mark_crashed)
                response = await page.goto(url, wait_until='networkidle')
                await page.wait_for_timeout(250)
                canvas_count = await page.locator('canvas').count()
                http_status = response.status if response is not None else 0
            finally:
                await browser.close()
        return BrowserObservation(
            http_status=http_status,
            canvas_count=canvas_count,
            console_errors=tuple(console_errors),
            page_errors=tuple(page_errors),
            crashed=crashed,
        )


def available_tcp_port(host: str = '127.0.0.1') -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((host, 0))
        return int(listener.getsockname()[1])


def _http_status(url: str) -> int | None:
    try:
        with urlopen(url, timeout=0.5) as response:
            return int(response.status)
    except HTTPError as error:
        return int(error.code)
    except (OSError, URLError):
        return None
