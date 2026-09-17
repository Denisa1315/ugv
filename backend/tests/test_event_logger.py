from app.modules.telemetry.event_logger import EventLogger


def test_log_and_retrieve(tmp_path):
    logger = EventLogger(str(tmp_path / "events.db"))
    logger.log("PATH_REPLANNED", "INFO", x=1.0, y=2.0, risk=0.3, confidence=0.8, description="test event")

    events = logger.recent()
    assert len(events) == 1
    assert events[0].event_type == "PATH_REPLANNED"
    assert events[0].severity == "INFO"
    assert events[0].position_x == 1.0
    assert events[0].description == "test event"


def test_recent_returns_newest_first(tmp_path):
    logger = EventLogger(str(tmp_path / "events.db"))
    for i in range(5):
        logger.log("PATH_REPLANNED", "INFO", x=float(i), y=0.0, risk=0.1, confidence=0.9, description=f"event {i}")

    events = logger.recent(limit=3)
    assert len(events) == 3
    assert [e.position_x for e in events] == [4.0, 3.0, 2.0]


def test_as_dict_shape(tmp_path):
    logger = EventLogger(str(tmp_path / "events.db"))
    event = logger.log("EMERGENCY_STOP", "CRITICAL", x=5.0, y=5.0, risk=0.9, confidence=0.1, description="too close")
    d = event.as_dict()
    assert d["event_type"] == "EMERGENCY_STOP"
    assert d["position"] == {"x": 5.0, "y": 5.0}
    assert d["severity"] == "CRITICAL"


def test_persists_across_logger_instances(tmp_path):
    db_path = str(tmp_path / "events.db")
    EventLogger(db_path).log("PATH_REPLANNED", "INFO", 0.0, 0.0, 0.0, 1.0, "first")
    second_logger = EventLogger(db_path)
    events = second_logger.recent()
    assert len(events) == 1
    assert events[0].description == "first"
