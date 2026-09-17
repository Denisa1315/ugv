from __future__ import annotations

import time

from app.db.models import EventLog
from app.db.session import make_session_factory


class EventLogger:
    def __init__(self, db_path: str = "ugv_events.db") -> None:
        self.session_factory = make_session_factory(db_path)

    def log(
        self,
        event_type: str,
        severity: str,
        x: float,
        y: float,
        risk: float,
        confidence: float,
        description: str,
    ) -> EventLog:
        with self.session_factory() as session:
            event = EventLog(
                timestamp=time.time(),
                event_type=event_type,
                severity=severity,
                position_x=x,
                position_y=y,
                risk=risk,
                confidence=confidence,
                description=description,
            )
            session.add(event)
            session.commit()
            session.refresh(event)
            return event

    def recent(self, limit: int = 50) -> list[EventLog]:
        with self.session_factory() as session:
            return list(
                session.query(EventLog).order_by(EventLog.id.desc()).limit(limit).all()
            )
