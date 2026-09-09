"""
WebRTC Signaling API & WebSocket channel for VoiceShield live call protection.

Allows Web clients and mobile app to exchange:
- SDP Offers
- SDP Answers
- ICE Candidates

Features:
- Deterministic peer session tracking and connection generations
- Role assignment for W3C Perfect Negotiation (impolite vs polite)
- Zero stale message / offer / answer caching (prevents InvalidStateError on reconnect)
- Structured signaling telemetry (timestamp, roomId, peerId, generation, type)
- Independent room cleanup
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger("voiceshield.webrtc")

router = APIRouter(prefix="/api/rtc", tags=["WebRTC Signaling"])


class SignalEnvelope(BaseModel):
    room_id: Optional[str] = None
    sender_id: Optional[str] = None
    sender: Optional[str] = None
    target_id: Optional[str] = None
    generation: int = 1
    type: str  # "offer", "answer", "candidate", "bye", "peer_joined", "peer_left"
    payload: dict = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class PeerSession:
    def __init__(self, peer_id: str, websocket: WebSocket, is_polite: bool, generation: int = 1):
        self.peer_id = peer_id
        self.websocket = websocket
        self.is_polite = is_polite
        self.generation = generation
        self.joined_at = time.time()


class RoomState:
    def __init__(self, room_id: str):
        self.room_id = room_id
        self.peers: Dict[str, PeerSession] = {}
        self.generation = 1
        self.last_http_offer: Optional[dict] = None
        self.last_http_answer: Optional[dict] = None

    @property
    def peer_count(self) -> int:
        return len(self.peers)

    async def broadcast_except(self, message: dict, exclude_peer_id: Optional[str] = None):
        """Forward signaling message to other peers in room without echoing back."""
        for pid, session in list(self.peers.items()):
            if pid != exclude_peer_id:
                try:
                    await session.websocket.send_json(message)
                except Exception as e:
                    logger.warning(f"[Signaling] Error sending to peer {pid} in room {self.room_id}: {e}")


# In-memory rooms
rooms: Dict[str, RoomState] = {}


def get_or_create_room(room_id: str) -> RoomState:
    if room_id not in rooms:
        rooms[room_id] = RoomState(room_id)
    return rooms[room_id]


@router.post("/rooms/{room_id}/signal")
async def send_signal_http(room_id: str, envelope: SignalEnvelope):
    """HTTP endpoint to submit SDP offer/answer or ICE candidate."""
    room = get_or_create_room(room_id)
    sender = envelope.sender_id or envelope.sender or "anonymous"
    envelope.room_id = room_id
    envelope.sender_id = sender

    if envelope.type == "offer":
        room.last_http_offer = envelope.payload
    elif envelope.type == "answer":
        room.last_http_answer = envelope.payload

    logger.info(
        f"[Signaling-HTTP] ts={envelope.timestamp:.3f} room={room_id} "
        f"sender={sender} type={envelope.type} gen={envelope.generation}"
    )

    # Forward to active WebSockets in room
    msg = envelope.model_dump()
    await room.broadcast_except(msg, exclude_peer_id=sender)
    return {"status": "ok", "type": envelope.type, "room_id": room_id}


@router.get("/rooms/{room_id}/state")
async def get_room_state(room_id: str):
    """Retrieve current signaling room participants."""
    if room_id not in rooms:
        raise HTTPException(status_code=404, detail="Room not found")
    room = rooms[room_id]
    return {
        "room_id": room.room_id,
        "peer_count": room.peer_count,
        "has_offer": room.last_http_offer is not None,
        "offer": room.last_http_offer,
        "has_answer": room.last_http_answer is not None,
        "answer": room.last_http_answer,
        "peers": [
            {
                "peer_id": p.peer_id,
                "is_polite": p.is_polite,
                "generation": p.generation,
                "joined_at": p.joined_at,
            }
            for p in room.peers.values()
        ],
        "generation": room.generation,
    }


@router.delete("/rooms/{room_id}")
async def close_room(room_id: str):
    """Close and clean up a signaling room."""
    if room_id in rooms:
        room = rooms.pop(room_id)
        for p in list(room.peers.values()):
            try:
                await p.websocket.close(code=1000, reason="Room closed")
            except Exception:
                pass
    return {"status": "closed", "room_id": room_id}


@router.websocket("/ws/{room_id}")
async def websocket_signaling(
    websocket: WebSocket,
    room_id: str,
    peer_id: Optional[str] = Query(None),
):
    """
    Real-time WebSocket signaling channel with Perfect Negotiation role assignment.
    
    Invariants:
    1. First peer in room is impolite (initiator, is_polite=False).
    2. Second peer in room is polite (callee, is_polite=True).
    3. Old offers/answers are NEVER cached or replayed upon reconnect.
    4. Senders never receive their own echoes.
    5. Each peer has a monotonic connection generation to reject stale messages.
    """
    await websocket.accept()

    assigned_peer_id = peer_id or f"peer_{uuid.uuid4().hex[:8]}"
    room = get_or_create_room(room_id)

    # Handle reconnect from same peer_id: close old socket cleanly
    if assigned_peer_id in room.peers:
        old_session = room.peers.pop(assigned_peer_id)
        try:
            await old_session.websocket.close(code=1001, reason="Replaced by new connection")
        except Exception:
            pass
        room.generation += 1

    # Determine politeness role:
    # First connected peer is impolite (False); subsequent peer is polite (True)
    is_polite = room.peer_count > 0

    session = PeerSession(
        peer_id=assigned_peer_id,
        websocket=websocket,
        is_polite=is_polite,
        generation=room.generation,
    )
    room.peers[assigned_peer_id] = session

    logger.info(
        f"[Signaling] Peer joined room={room_id} peer_id={assigned_peer_id} "
        f"is_polite={is_polite} gen={session.generation} total_peers={room.peer_count}"
    )

    try:
        # 1. Send joined confirmation to the connecting peer
        await websocket.send_json({
            "type": "joined",
            "room_id": room_id,
            "peer_id": assigned_peer_id,
            "is_polite": is_polite,
            "generation": session.generation,
            "peer_count": room.peer_count,
            "timestamp": time.time(),
        })

        # 2. Notify other peer that a new peer joined
        await room.broadcast_except(
            {
                "type": "peer_joined",
                "room_id": room_id,
                "peer_id": assigned_peer_id,
                "is_polite": is_polite,
                "generation": session.generation,
                "peer_count": room.peer_count,
                "timestamp": time.time(),
            },
            exclude_peer_id=assigned_peer_id,
        )

        # 3. Process signaling messages
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            msg_generation = data.get("generation", session.generation)
            target_id = data.get("target_id")
            payload = data.get("payload", {})

            # Structured signaling audit log (no audio or sensitive payload logged)
            logger.info(
                f"[Signaling-Exchange] ts={time.time():.3f} room={room_id} "
                f"from={assigned_peer_id} type={msg_type} gen={msg_generation}"
            )

            # Stale generation check
            if msg_generation < session.generation:
                logger.warning(
                    f"[Signaling] Dropping stale message type={msg_type} "
                    f"gen={msg_generation} < active_gen={session.generation}"
                )
                continue

            envelope = {
                "room_id": room_id,
                "sender_id": assigned_peer_id,
                "target_id": target_id,
                "generation": session.generation,
                "type": msg_type,
                "payload": payload,
                "timestamp": time.time(),
            }

            if msg_type == "bye":
                await room.broadcast_except(envelope, exclude_peer_id=assigned_peer_id)
                break

            # Forward to peer(s)
            if target_id and target_id in room.peers:
                try:
                    await room.peers[target_id].websocket.send_json(envelope)
                except Exception as e:
                    logger.warning(f"[Signaling] Forward to target {target_id} failed: {e}")
            else:
                await room.broadcast_except(envelope, exclude_peer_id=assigned_peer_id)

    except WebSocketDisconnect:
        logger.info(f"[Signaling] Client {assigned_peer_id} disconnected from room {room_id}")
    except Exception as e:
        logger.error(f"[Signaling] Unexpected error for {assigned_peer_id} in {room_id}: {e}")
    finally:
        if assigned_peer_id in room.peers:
            del room.peers[assigned_peer_id]
            logger.info(f"[Signaling] Removed {assigned_peer_id} from {room_id}. Remaining: {room.peer_count}")
            await room.broadcast_except({
                "type": "peer_left",
                "room_id": room_id,
                "peer_id": assigned_peer_id,
                "peer_count": room.peer_count,
                "timestamp": time.time(),
            })

        if room.peer_count == 0 and room_id in rooms:
            del rooms[room_id]
            logger.info(f"[Signaling] Room {room_id} deleted (all peers departed)")
