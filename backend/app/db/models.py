from __future__ import annotations

from sqlalchemy import Float, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class EventLog(Base):
    """One row per state-changing event — not a per-tick telemetry dump.
    Real-time frame-by-frame state is what the WebSocket stream is for;
    this table is the durable, queryable "what happened and when" record."""

    __tablename__ = "event_log"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timestamp: Mapped[float] = mapped_column(Float)
    event_type: Mapped[str] = mapped_column(String(64))  # e.g. PATH_REPLANNED, DEGRADED_ENTERED, EMERGENCY_STOP
    severity: Mapped[str] = mapped_column(String(16))  # INFO | WARNING | CRITICAL
    position_x: Mapped[float] = mapped_column(Float)
    position_y: Mapped[float] = mapped_column(Float)
    risk: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    description: Mapped[str] = mapped_column(String(500))

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "severity": self.severity,
            "position": {"x": self.position_x, "y": self.position_y},
            "risk": self.risk,
            "confidence": self.confidence,
            "description": self.description,
        }
