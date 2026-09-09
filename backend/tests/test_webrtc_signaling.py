import asyncio
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_webrtc_rest_signaling():
    room_id = "test-room-rest-1"
    
    # 1. Post Offer
    offer_payload = {"sdp": "v=0\r\no=- 12345 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n", "type": "offer"}
    res = client.post(f"/api/rtc/rooms/{room_id}/signal", json={
        "sender": "peer_a",
        "type": "offer",
        "payload": offer_payload
    })
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    
    # 2. Check room state
    state_res = client.get(f"/api/rtc/rooms/{room_id}/state")
    assert state_res.status_code == 200
    state = state_res.json()
    assert state["has_offer"] is True
    assert state["offer"] == offer_payload
    assert state["has_answer"] is False
    
    # 3. Post Answer
    answer_payload = {"sdp": "v=0\r\no=- 67890 2 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n", "type": "answer"}
    res2 = client.post(f"/api/rtc/rooms/{room_id}/signal", json={
        "sender": "peer_b",
        "type": "answer",
        "payload": answer_payload
    })
    assert res2.status_code == 200
    
    # 4. Check updated room state
    state_res2 = client.get(f"/api/rtc/rooms/{room_id}/state")
    state2 = state_res2.json()
    assert state2["has_answer"] is True
    assert state2["answer"] == answer_payload

    # 5. Clean up room
    del_res = client.delete(f"/api/rtc/rooms/{room_id}")
    assert del_res.status_code == 200


def test_webrtc_websocket_signaling():
    room_id = "test-room-ws-1"

    # Connect Peer A and Peer B over WebSocket
    with client.websocket_connect(f"/api/rtc/ws/{room_id}") as ws_a:
        ws_a.receive_json()  # joined handshake
        with client.websocket_connect(f"/api/rtc/ws/{room_id}") as ws_b:
            ws_b.receive_json()  # joined handshake
            ws_a.receive_json()  # peer_joined handshake notification

            # Peer A sends offer
            offer_msg = {
                "type": "offer",
                "payload": {"sdp": "mock-sdp-offer", "type": "offer"}
            }
            ws_a.send_json(offer_msg)

            # Peer B receives offer
            received_by_b = ws_b.receive_json()
            assert received_by_b["type"] == "offer"
            assert received_by_b["payload"]["sdp"] == "mock-sdp-offer"

            # Peer B sends answer
            answer_msg = {
                "type": "answer",
                "payload": {"sdp": "mock-sdp-answer", "type": "answer"}
            }
            ws_b.send_json(answer_msg)

            # Peer A receives answer
            received_by_a = ws_a.receive_json()
            assert received_by_a["type"] == "answer"
            assert received_by_a["payload"]["sdp"] == "mock-sdp-answer"

            # Peer A sends ICE candidate
            candidate_msg = {
                "type": "candidate",
                "payload": {"candidate": "candidate:1 1 UDP 2122260223 ...", "sdpMid": "0", "sdpMLineIndex": 0}
            }
            ws_a.send_json(candidate_msg)

            # Peer B receives candidate
            cand_received = ws_b.receive_json()
            assert cand_received["type"] == "candidate"
            assert cand_received["payload"]["candidate"] == candidate_msg["payload"]["candidate"]

    print("WebRTC WebSocket 2-party signaling verified successfully!")


if __name__ == "__main__":
    test_webrtc_rest_signaling()
    test_webrtc_websocket_signaling()
    print("ALL WEBRTC SIGNALING TESTS PASSED!")
