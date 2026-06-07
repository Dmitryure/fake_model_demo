from __future__ import annotations

import logging

import uvicorn

from .settings import DEFAULT_HOST, DEFAULT_PORT


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "backend.app:app",
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
