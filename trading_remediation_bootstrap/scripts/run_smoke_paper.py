#!/usr/bin/env python3
"""Minimal paper-trading smoke harness placeholder.
Replace stub functions with your real data/strategy/order integrations.
"""
# Standard library imports
import random
import time

SYMBOLS = ["SPY", "AAPL"]
MAX_RISK_PER_TRADE = 0.01  # 1% of notional (example)


def fetch_prices(symbols: list[str]) -> dict[str, float]:
    # TODO: replace with real adapter (e.g., yfinance)
    return {s: 100 + random.random() for s in symbols}


def simple_strategy(prices: dict[str, float]) -> list[dict]:
    # TODO: replace with real signals. Here: random no-op.
    decisions = []
    for s, p in prices.items():
        if random.random() < 0.1:
            decisions.append({"symbol": s, "side": "BUY", "qty": 1})
    return decisions


def risk_check(order: dict) -> bool:
    # TODO: replace with real pre-trade checks
    return True


def send_order_paper(order: dict) -> bool:
    # TODO: send to paper broker; here we simulate success
    time.sleep(0.05)
    return True


def main():
    prices = fetch_prices(SYMBOLS)
    decisions = simple_strategy(prices)
    ok = True
    for d in decisions:
        if risk_check(d):
            ok = ok and send_order_paper(d)
    print("SMOKE PASS" if ok else "SMOKE FAIL")


if __name__ == "__main__":
    main()
