"""CLI entry point for Coinbase LOB data recording.

Usage:
    python -m lob_forge.data.coinbase_downloader [--duration SECONDS] [--output OUTPUT_DIR]

Records real-time BTC-USD LOB data via Coinbase WebSocket and saves to Parquet.
"""

from __future__ import annotations

import argparse
import logging

from lob_forge.data.downloader import CoinbaseDownloader

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record Coinbase BTC-USD LOB data")
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Recording duration in seconds (default 300)",
    )
    parser.add_argument(
        "--output", type=str, default="data/", help="Output directory (default data/)"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC-USD",
        help="Coinbase product ID (default BTC-USD)",
    )
    parser.add_argument(
        "--depth", type=int, default=10, help="LOB depth levels (default 10)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    downloader = CoinbaseDownloader(
        symbol=args.symbol,
        depth=args.depth,
        output_dir=args.output,
    )

    logger.info(
        "Starting Coinbase LOB recording: %s for %ds", args.symbol, args.duration
    )
    out_path = downloader.record_websocket(duration_seconds=args.duration)
    logger.info("Recording complete: %s", out_path)


if __name__ == "__main__":
    main()
