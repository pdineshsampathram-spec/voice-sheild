import { NextRequest, NextResponse } from "next/server";

interface SignalingMessage {
  id: number;
  sender_id: string;
  type: string;
  payload: any;
  timestamp: number;
}

interface RoomRecord {
  roomId: string;
  peers: Set<string>;
  impolitePeer: string | null;
  messages: SignalingMessage[];
  seq: number;
  lastActive: number;
}

// In-memory room store (global across warm serverless invocations)
const rooms = new Map<string, RoomRecord>();

function getOrCreateRoom(roomId: string): RoomRecord {
  let room = rooms.get(roomId);
  if (!room) {
    room = {
      roomId,
      peers: new Set<string>(),
      impolitePeer: null,
      messages: [],
      seq: 0,
      lastActive: Date.now(),
    };
    rooms.set(roomId, room);
  }
  // Clean up rooms older than 1 hour
  if (rooms.size > 200) {
    const cutoff = Date.now() - 3600000;
    for (const [id, r] of rooms.entries()) {
      if (r.lastActive < cutoff) rooms.delete(id);
    }
  }
  room.lastActive = Date.now();
  return room;
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { action, room_id, peer_id, message, last_seq = 0 } = body;

    if (!room_id || !peer_id) {
      return NextResponse.json({ error: "Missing room_id or peer_id" }, { status: 400 });
    }

    const room = getOrCreateRoom(room_id);

    if (action === "join") {
      room.peers.add(peer_id);
      if (!room.impolitePeer) {
        room.impolitePeer = peer_id;
      }
      const isPolite = room.impolitePeer !== peer_id;
      const count = room.peers.size;

      // Broadcast peer_joined if count > 1
      if (count > 1) {
        room.seq++;
        room.messages.push({
          id: room.seq,
          sender_id: peer_id,
          type: "peer_joined",
          payload: { peer_count: count },
          timestamp: Date.now() / 1000,
        });
      }

      return NextResponse.json({
        type: "joined",
        is_polite: isPolite,
        peer_count: count,
        generation: 1,
        current_seq: room.seq,
      });
    }

    if (action === "send") {
      room.seq++;
      const envelope: SignalingMessage = {
        id: room.seq,
        sender_id: peer_id,
        type: message.type,
        payload: message.payload,
        timestamp: Date.now() / 1000,
      };
      room.messages.push(envelope);
      // Keep last 50 messages
      if (room.messages.length > 50) {
        room.messages = room.messages.slice(-50);
      }
      return NextResponse.json({ status: "sent", seq: room.seq });
    }

    if (action === "poll") {
      room.peers.add(peer_id);
      const incoming = room.messages.filter(
        (m) => m.id > last_seq && m.sender_id !== peer_id
      );
      return NextResponse.json({
        messages: incoming,
        current_seq: room.seq,
        peer_count: room.peers.size,
      });
    }

    if (action === "leave") {
      room.peers.delete(peer_id);
      if (room.impolitePeer === peer_id) {
        room.impolitePeer = room.peers.size > 0 ? Array.from(room.peers)[0] : null;
      }
      room.seq++;
      room.messages.push({
        id: room.seq,
        sender_id: peer_id,
        type: "peer_left",
        payload: { peer_count: room.peers.size },
        timestamp: Date.now() / 1000,
      });
      return NextResponse.json({ status: "left", peer_count: room.peers.size });
    }

    return NextResponse.json({ error: "Unknown action" }, { status: 400 });
  } catch (err: any) {
    return NextResponse.json({ error: err.message || "Signaling failure" }, { status: 500 });
  }
}

