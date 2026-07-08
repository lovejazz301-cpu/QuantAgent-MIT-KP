"""
Paper-trading engine for QuantAgent's AI Trade feature.

Tracks a simulated portfolio and auto-opens positions whenever the
multi-agent pipeline issues a LONG/SHORT decision. No real money or
broker connection is involved -- everything is tracked against a
virtual cash balance persisted to a JSON file.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

STARTING_BALANCE = 100_000.0
RISK_PER_TRADE = 0.02  # fraction of balance risked per trade
STOP_DISTANCE_PCT = 0.015  # default stop-loss distance from entry
DEFAULT_RISK_REWARD = 1.5
MIN_RISK_REWARD = 0.5
MAX_RISK_REWARD = 5.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_risk_reward(raw: Any) -> float:
    """Parse a risk/reward ratio from the decision agent's output.

    Accepts a bare float ("1.5"), a ratio string ("1:2.5"), or a
    numeric type. Falls back to DEFAULT_RISK_REWARD on anything else.
    """
    try:
        if raw is None:
            return DEFAULT_RISK_REWARD
        if isinstance(raw, (int, float)):
            value = float(raw)
        else:
            text = str(raw).strip()
            match = re.match(r"^\s*([\d.]+)\s*:\s*([\d.]+)\s*$", text)
            if match:
                left, right = float(match.group(1)), float(match.group(2))
                value = right / left if left else DEFAULT_RISK_REWARD
            else:
                value = float(re.search(r"[\d.]+", text).group())
        return max(MIN_RISK_REWARD, min(MAX_RISK_REWARD, value))
    except (ValueError, AttributeError, TypeError):
        return DEFAULT_RISK_REWARD


class TradeEngine:
    """Manages a simulated paper-trading portfolio."""

    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.state = self._load()

    def _default_state(self) -> Dict[str, Any]:
        return {
            "balance": STARTING_BALANCE,
            "starting_balance": STARTING_BALANCE,
            "next_id": 1,
            "open_positions": [],
            "closed_positions": [],
        }

    def _load(self) -> Dict[str, Any]:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "balance" in data:
                        return data
            except (json.JSONDecodeError, OSError):
                pass
        return self._default_state()

    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def reset(self) -> None:
        self.state = self._default_state()
        self._save()

    def _unrealized_pnl(self, position: Dict[str, Any], price: float) -> float:
        qty = position["quantity"]
        if position["direction"] == "LONG":
            return (price - position["entry_price"]) * qty
        return (position["entry_price"] - price) * qty

    def auto_execute(
        self,
        asset: str,
        symbol: str,
        decision: str,
        current_price: float,
        risk_reward_ratio: Any,
        timeframe: str,
    ) -> Dict[str, Any]:
        """Open a paper position for a LONG/SHORT decision, if not already open on this symbol."""
        if decision not in ("LONG", "SHORT"):
            return {"executed": False, "reason": f"No auto-trade for decision '{decision}'"}

        if not current_price or current_price <= 0:
            return {"executed": False, "reason": "Invalid current price"}

        for pos in self.state["open_positions"]:
            if pos["symbol"] == symbol:
                return {
                    "executed": False,
                    "reason": f"Position already open for {asset} ({pos['direction']})",
                }

        rr = parse_risk_reward(risk_reward_ratio)
        stop_distance = current_price * STOP_DISTANCE_PCT
        risk_amount = self.state["balance"] * RISK_PER_TRADE
        quantity = risk_amount / stop_distance if stop_distance else 0

        if quantity <= 0:
            return {"executed": False, "reason": "Computed position size was zero"}

        if decision == "LONG":
            stop_loss = current_price - stop_distance
            take_profit = current_price + stop_distance * rr
        else:
            stop_loss = current_price + stop_distance
            take_profit = current_price - stop_distance * rr

        position = {
            "id": self.state["next_id"],
            "asset": asset,
            "symbol": symbol,
            "direction": decision,
            "entry_price": current_price,
            "quantity": quantity,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward_ratio": rr,
            "timeframe": timeframe,
            "opened_at": _now_iso(),
            "last_price": current_price,
            "status": "open",
        }
        self.state["next_id"] += 1
        self.state["open_positions"].append(position)
        self._save()

        return {"executed": True, "position": position}

    def update_prices(self, price_map: Dict[str, float]) -> None:
        """Mark open positions to market and auto-close any that hit stop-loss/take-profit."""
        still_open = []
        for pos in self.state["open_positions"]:
            price = price_map.get(pos["symbol"])
            if price is None or price <= 0:
                still_open.append(pos)
                continue

            pos["last_price"] = price

            hit_stop = (
                (pos["direction"] == "LONG" and price <= pos["stop_loss"])
                or (pos["direction"] == "SHORT" and price >= pos["stop_loss"])
            )
            hit_target = (
                (pos["direction"] == "LONG" and price >= pos["take_profit"])
                or (pos["direction"] == "SHORT" and price <= pos["take_profit"])
            )

            if hit_stop or hit_target:
                exit_price = pos["stop_loss"] if hit_stop else pos["take_profit"]
                reason = "stop_loss" if hit_stop else "take_profit"
                self._close(pos, exit_price, reason)
            else:
                still_open.append(pos)

        self.state["open_positions"] = still_open
        self._save()

    def _close(self, position: Dict[str, Any], exit_price: float, reason: str) -> Dict[str, Any]:
        pnl = self._unrealized_pnl(position, exit_price)
        notional = position["entry_price"] * position["quantity"]
        pnl_pct = (pnl / notional * 100) if notional else 0.0

        closed = dict(position)
        closed.update(
            {
                "exit_price": exit_price,
                "closed_at": _now_iso(),
                "close_reason": reason,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "status": "closed",
            }
        )
        self.state["balance"] += pnl
        self.state["closed_positions"].append(closed)
        return closed

    def close_position(self, position_id: int, exit_price: float, reason: str = "manual") -> Optional[Dict[str, Any]]:
        for i, pos in enumerate(self.state["open_positions"]):
            if pos["id"] == position_id:
                closed = self._close(pos, exit_price, reason)
                del self.state["open_positions"][i]
                self._save()
                return closed
        return None

    def get_state(self) -> Dict[str, Any]:
        unrealized_total = sum(
            self._unrealized_pnl(pos, pos["last_price"]) for pos in self.state["open_positions"]
        )
        realized_total = self.state["balance"] - self.state["starting_balance"]
        closed = self.state["closed_positions"]
        wins = [p for p in closed if p["pnl"] > 0]
        win_rate = (len(wins) / len(closed) * 100) if closed else 0.0

        open_positions = []
        for pos in self.state["open_positions"]:
            p = dict(pos)
            p["unrealized_pnl"] = self._unrealized_pnl(pos, pos["last_price"])
            notional = pos["entry_price"] * pos["quantity"]
            p["unrealized_pnl_pct"] = (p["unrealized_pnl"] / notional * 100) if notional else 0.0
            open_positions.append(p)

        return {
            "balance": self.state["balance"],
            "starting_balance": self.state["starting_balance"],
            "equity": self.state["balance"] + unrealized_total,
            "unrealized_pnl": unrealized_total,
            "realized_pnl": realized_total,
            "open_positions": sorted(open_positions, key=lambda p: p["opened_at"], reverse=True),
            "closed_positions": sorted(closed, key=lambda p: p["closed_at"], reverse=True),
            "total_trades": len(closed),
            "win_rate": win_rate,
        }
