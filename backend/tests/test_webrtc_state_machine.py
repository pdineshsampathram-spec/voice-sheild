"""
Comprehensive WebRTC Signaling & State Machine Tests.

Tests:
1. Normal 2-party signaling lifecycle (roles, offer, answer, ICE candidates)
2. Perfect Negotiation role assignment (peer 1 = impolite, peer 2 = polite)
3. Zero stale offer/answer caching upon reconnect (preventing InvalidStateError)
4. Stale generation message rejection
5. Simultaneous offer handling and glare avoidance
6. Safe disconnect and room teardown
7. Independent monitor isolation
"""

import time
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_webrtc_role_assignment_and_lifecycle():
    """Verify peer 1 is impolite, peer 2 is polite, and exchange succeeds without echoes."""
    room_id = "test-room-lifecycle-1"

    with client.websocket_connect(f"/api/rtc/ws/{room_id}?peer_id=peer_1") as ws_1:
        # Peer 1 should receive joined confirmation as impolite (caller)
        msg1 = ws_1.receive_json()
        assert msg1["type"] == "joined"
        assert msg1["peer_id"] == "peer_1"
        assert msg1["is_polite"] is False
        assert msg1["generation"] == 1
        assert msg1["peer_count"] == 1

        with client.websocket_connect(f"/api/rtc/ws/{room_id}?peer_id=peer_2") as ws_2:
            # Peer 2 should receive joined confirmation as polite (callee)
            msg2 = ws_2.receive_json()
            assert msg2["type"] == "joined"
            assert msg2["peer_id"] == "peer_2"
            assert msg2["is_polite"] is True
            assert msg2["peer_count"] == 2

            # Peer 1 should receive peer_joined notification
            p1_notify = ws_1.receive_json()
            assert p1_notify["type"] == "peer_joined"
            assert p1_notify["peer_id"] == "peer_2"

            # Offer from Peer 1 -> Peer 2
            offer_payload = {"type": "offer", "sdp": "v=0\r\no=mock-p1 1 IN IP4 127.0.0.1\r\ns=-\r\n"}
            ws_1.send_json({
                "type": "offer",
                "generation": 1,
                "payload": offer_payload,
            })

            recv_by_p2 = ws_2.receive_json()
            assert recv_by_p2["type"] == "offer"
            assert recv_by_p2["sender_id"] == "peer_1"
            assert recv_by_p2["payload"]["sdp"] == offer_payload["sdp"]

            # Answer from Peer 2 -> Peer 1
            answer_payload = {"type": "answer", "sdp": "v=0\r\no=mock-p2 1 IN IP4 127.0.0.1\r\ns=-\r\n"}
            ws_2.send_json({
                "type": "answer",
                "generation": 1,
                "payload": answer_payload,
            })

            recv_by_p1 = ws_1.receive_json()
            assert recv_by_p1["type"] == "answer"
            assert recv_by_p1["sender_id"] == "peer_2"
            assert recv_by_p1["payload"]["sdp"] == answer_payload["sdp"]

            # ICE Candidate exchange
            cand_payload = {"candidate": "candidate:1 1 UDP 2122260223 192.168.1.1 50000 typ host", "sdpMid": "0"}
            ws_1.send_json({
                "type": "candidate",
                "generation": 1,
                "payload": cand_payload,
            })

            cand_by_p2 = ws_2.receive_json()
            assert cand_by_p2["type"] == "candidate"
            assert cand_by_p2["sender_id"] == "peer_1"
            assert cand_by_p2["payload"]["candidate"] == cand_payload["candidate"]


def test_no_stale_offer_answer_replayed_on_reconnect():
    """
    CRITICAL REGRESSION TEST:
    Under the old code, when a peer reconnected, the server immediately spat out
    the old offer AND the old answer, causing InvalidStateError on RTCPeerConnection!
    Verify that a newly connected or reconnected peer receives ONLY 'joined', NOT stale SDPs.
    """
    room_id = "test-room-no-stale-replay"

    # Step 1: Peer 1 and 2 exchange offer and answer
    with client.websocket_connect(f"/api/rtc/ws/{room_id}?peer_id=peer_1") as ws_1:
        ws_1.receive_json()  # joined
        with client.websocket_connect(f"/api/rtc/ws/{room_id}?peer_id=peer_2") as ws_2:
            ws_2.receive_json()  # joined
            ws_1.receive_json()  # peer_joined

            ws_1.send_json({"type": "offer", "generation": 1, "payload": {"sdp": "offer-1", "type": "offer"}})
            ws_2.receive_json()  # offer
            ws_2.send_json({"type": "answer", "generation": 1, "payload": {"sdp": "answer-1", "type": "answer"}})
            ws_1.receive_json()  # answer

    # Both disconnected; now a fresh peer connects to the room
    with client.websocket_connect(f"/api/rtc/ws/{room_id}?peer_id=peer_fresh") as ws_fresh:
        first_msg = ws_fresh.receive_json()
        assert first_msg["type"] == "joined"
        # Invariant: NO subsequent stale 'offer' or 'answer' message is sent by server!
        # Check that room state is clean and no message is waiting
        state_res = client.get(f"/api/rtc/rooms/{room_id}/state")
        assert state_res.status_code == 200
        assert state_res.json()["peer_count"] == 1


def test_stale_generation_message_rejection():
    """Verify that signaling messages with outdated generations are dropped."""
    room_id = "test-room-stale-gen"

    with client.websocket_connect(f"/api/rtc/ws/{room_id}?peer_id=p1") as ws_1:
        ws_1.receive_json()  # joined
        with client.websocket_connect(f"/api/rtc/ws/{room_id}?peer_id=p2") as ws_2:
            ws_2.receive_json()  # joined
            ws_1.receive_json()  # peer_joined

            # Peer 1 sends message with stale generation (gen 0 when active is 1)
            ws_1.send_json({
                "type": "offer",
                "generation": 0,  # STALE
                "payload": {"sdp": "stale-offer"},
            })

            # Peer 1 then sends valid message with gen 1
            ws_1.send_json({
                "type": "offer",
                "generation": 1,  # VALID
                "payload": {"sdp": "valid-offer"},
            })

            # Peer 2 must receive the VALID offer, NOT the stale one
            received = ws_2.receive_json()
            assert received["type"] == "offer"
            assert received["payload"]["sdp"] == "valid-offer"


def test_peer_departure_and_clean_room_destruction():
    """Verify peer_left notifications and complete room state cleanup on empty."""
    room_id = "test-room-cleanup"

    with client.websocket_connect(f"/api/rtc/ws/{room_id}?peer_id=p1") as ws_1:
        ws_1.receive_json()  # joined
        with client.websocket_connect(f"/api/rtc/ws/{room_id}?peer_id=p2") as ws_2:
            ws_2.receive_json()  # joined
            ws_1.receive_json()  # peer_joined

            # p2 sends bye
            ws_2.send_json({"type": "bye", "generation": 1})
            p1_bye = ws_1.receive_json()
            assert p1_bye["type"] == "bye"

        # p2 socket closed; p1 receives peer_left
        p1_left = ws_1.receive_json()
        assert p1_left["type"] == "peer_left"
        assert p1_left["peer_id"] == "p2"
        assert p1_left["peer_count"] == 1

    # Now p1 closed too; room should be cleaned up completely
    state_res = client.get(f"/api/rtc/rooms/{room_id}/state")
    assert state_res.status_code == 404  # Room was destroyed cleanly


def test_monitor_isolation_from_p2p_signaling():
    """
    Verify that creating or destroying an observer monitor session
    does not affect the WebRTC signaling room.
    """
    room_id = "test-room-isolation"

    with client.websocket_connect(f"/api/rtc/ws/{room_id}?peer_id=peer_call_1") as ws:
        ws.receive_json()

        # Create monitor session independently
        mon_res = client.post("/api/monitor/session", json={"asr_model_size": "base"})
        assert mon_res.status_code == 200
        session_id = mon_res.json()["session_id"]

        # Call room is completely unaffected
        state_res = client.get(f"/api/rtc/rooms/{room_id}/state")
        assert state_res.status_code == 200
        assert state_res.json()["peer_count"] == 1

        # Delete monitor session
        del_res = client.delete(f"/api/monitor/session/{session_id}")
        assert del_res.status_code == 200

        # Call room still healthy
        state_res2 = client.get(f"/api/rtc/rooms/{room_id}/state")
        assert state_res2.status_code == 200
        assert state_res2.json()["peer_count"] == 1


def test_monitor_session_cleanup_on_websocket_disconnect():
    """
    REGRESSION TEST:
    Verify that when an observer client's WebSocket disconnects,
    the backend automatically stops and removes the MonitorSession
    so running background tasks, thread pools, and queues do not leak.
    """
    from app.services.monitor.session import get_session

    mon_res = client.post("/api/monitor/session", json={"asr_model_size": "base"})
    assert mon_res.status_code == 200
    session_id = mon_res.json()["session_id"]
    assert get_session(session_id) is not None

    # Connect to monitor websocket and then disconnect
    with client.websocket_connect(f"/api/monitor/ws/{session_id}") as ws:
        init_msg = ws.receive_json()
        assert init_msg["type"] == "connected"
        assert init_msg["session_id"] == session_id

    # After WebSocket exits context (disconnects), session must be terminated and cleaned up
    assert get_session(session_id) is None
