from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping


SYMBOL_PATTERN = re.compile(r"^\d{6}\.(SH|SZ|BJ)$")
UNIVERSE_SPLIT_PATTERN = re.compile(r"[\s,，;；]+")


class InputValidationError(ValueError):
    pass


def normalize_symbol(value: Any) -> str:
    symbol = str(value or "").strip().upper()
    if not SYMBOL_PATTERN.fullmatch(symbol):
        raise InputValidationError(
            f"股票代码“{symbol or '(空)'}”格式不正确，"
            "请使用 600519.SH、000001.SZ 或 430047.BJ。"
        )
    return symbol


def parse_universe(value: str) -> list[str]:
    raw_symbols = [
        item
        for item in UNIVERSE_SPLIT_PATTERN.split(value.strip())
        if item
    ]
    if not raw_symbols:
        raise InputValidationError("股票池不能为空。")
    if len(raw_symbols) > 200:
        raise InputValidationError("股票池最多支持 200 只股票。")
    symbols = [normalize_symbol(item) for item in raw_symbols]
    duplicates = sorted(
        symbol for symbol in set(symbols) if symbols.count(symbol) > 1
    )
    if duplicates:
        raise InputValidationError(
            f"股票池存在重复代码：{', '.join(duplicates)}。"
        )
    return symbols


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def _decimal(value: Any, *, field: str, minimum: Decimal) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise InputValidationError(f"{field}必须是有效数字。") from exc
    if not number.is_finite() or number < minimum:
        raise InputValidationError(f"{field}不能小于 {minimum}。")
    return number


def _integer(
    value: Any,
    *,
    field: str,
    minimum: int,
) -> int:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise InputValidationError(f"{field}必须是整数。") from exc
    if not number.is_finite() or number != number.to_integral_value():
        raise InputValidationError(f"{field}必须是整数。")
    output = int(number)
    if output < minimum:
        raise InputValidationError(f"{field}不能小于 {minimum}。")
    return output


def parse_positions(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    positions: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=1):
        symbol_value = row.get("symbol")
        if _is_blank(symbol_value):
            other_values = [
                row.get("shares"),
                row.get("available_shares"),
                row.get("average_cost"),
                row.get("holding_days"),
            ]
            if all(_is_blank(item) for item in other_values):
                continue
            raise InputValidationError(f"持仓第 {row_number} 行缺少股票代码。")

        symbol = normalize_symbol(symbol_value)
        shares = _integer(
            row.get("shares"),
            field=f"{symbol} 持有股数",
            minimum=1,
        )
        available_value = row.get("available_shares")
        available_shares = (
            shares
            if _is_blank(available_value)
            else _integer(
                available_value,
                field=f"{symbol} 可卖股数",
                minimum=0,
            )
        )
        if available_shares > shares:
            raise InputValidationError(
                f"{symbol} 的可卖股数不能超过持有股数。"
            )
        average_cost = _decimal(
            row.get("average_cost"),
            field=f"{symbol} 平均成本",
            minimum=Decimal("0"),
        )
        holding_days_value = row.get("holding_days")
        holding_days = (
            None
            if _is_blank(holding_days_value)
            else _integer(
                holding_days_value,
                field=f"{symbol} 持有天数",
                minimum=0,
            )
        )
        positions.append(
            {
                "symbol": symbol,
                "shares": shares,
                "available_shares": available_shares,
                "average_cost": str(average_cost),
                "holding_days": holding_days,
            }
        )

    symbols = [position["symbol"] for position in positions]
    duplicates = sorted(
        symbol for symbol in set(symbols) if symbols.count(symbol) > 1
    )
    if duplicates:
        raise InputValidationError(
            f"持仓存在重复股票：{', '.join(duplicates)}。"
        )
    return positions
