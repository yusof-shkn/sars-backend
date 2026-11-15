from fastapi import APIRouter

from app.modules.events import service
from app.modules.events.schemas import Event, EventSummary, Probe

router = APIRouter(prefix="/events", tags=["events"])


@router.get("", response_model=list[EventSummary])
async def list_events() -> list[EventSummary]:
    return service.list_events()


@router.get("/{event_id}", response_model=Event)
async def get_event(event_id: str) -> Event:
    return service.get_event(event_id)


@router.get("/{event_id}/probe", response_model=Probe)
async def probe(event_id: str, lat: float, lon: float) -> Probe:
    return service.probe(event_id, lat, lon)
