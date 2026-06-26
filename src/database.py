import json
import time
import aiosqlite
from datetime import datetime, timezone

from src.config import DB_PATH
from src.models.schemas import ClaimObject


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_uuid TEXT PRIMARY KEY,
                video_url    TEXT NOT NULL,
                created_at   TIMESTAMP NOT NULL
            )
        """)
        for col in ("timeline_json TEXT", "duration_seconds INTEGER"):
            try:
                await db.execute(f"ALTER TABLE sessions ADD COLUMN {col}")
            except Exception:
                pass
        await db.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                claim_id                  TEXT PRIMARY KEY,
                session_uuid              TEXT NOT NULL REFERENCES sessions(session_uuid),
                timestamp_start           INTEGER,
                timestamp_end             INTEGER,
                claim_text                TEXT,
                importance_score          INTEGER DEFAULT 0,
                verification_status       TEXT,
                composite_confidence_score REAL,
                verdict_label             TEXT,
                explanation_summary       TEXT
            )
        """)
        try:
            await db.execute("ALTER TABLE claims ADD COLUMN updated_at REAL")
        except Exception:
            pass
        await db.commit()


async def create_session(session_uuid: str, video_url: str, duration_seconds: int = 0):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO sessions (session_uuid, video_url, created_at, duration_seconds) VALUES (?, ?, ?, ?)",
            (session_uuid, video_url, datetime.now(timezone.utc).isoformat(), duration_seconds),
        )
        await db.commit()


# Legacy batch insert used by old pipeline; kept for backward compat.
async def insert_claims(claims: list[ClaimObject], session_uuid: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany(
            """
            INSERT INTO claims
                (claim_id, session_uuid, timestamp_start, timestamp_end,
                 claim_text, importance_score, verification_status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    c.claim_id,
                    session_uuid,
                    c.timestamp_start,
                    c.timestamp_end,
                    c.claim_text,
                    c.importance_score,
                    c.verification_status,
                )
                for c in claims
            ],
        )
        await db.commit()


async def insert_single_claim(
    claim_id: str,
    session_uuid: str,
    claim_text: str,
    timestamp_start: int,
):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT OR IGNORE INTO claims
                (claim_id, session_uuid, timestamp_start, timestamp_end,
                 claim_text, importance_score, verification_status)
            VALUES (?, ?, ?, ?, ?, 0, 'PENDING')
            """,
            (claim_id, session_uuid, timestamp_start, timestamp_start, claim_text),
        )
        await db.commit()


async def update_claim_status(claim_id: str, status: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE claims SET verification_status = ? WHERE claim_id = ?",
            (status, claim_id),
        )
        await db.commit()


async def finalize_claim_verdict(claim_id: str, confidence: float, label: str, summary: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            UPDATE claims
            SET composite_confidence_score = ?,
                verdict_label              = ?,
                explanation_summary        = ?,
                verification_status        = 'VERIFIED',
                updated_at                 = ?
            WHERE claim_id = ?
            """,
            (confidence, label, summary, time.time(), claim_id),
        )
        await db.commit()


async def store_session_timeline(session_uuid: str, timeline_json: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE sessions SET timeline_json = ? WHERE session_uuid = ?",
            (timeline_json, session_uuid),
        )
        await db.commit()


async def get_verified_claims(session_uuid: str) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM claims WHERE session_uuid = ? AND verification_status = 'VERIFIED'",
            (session_uuid,),
        ) as cursor:
            rows = await cursor.fetchall()

    result = []
    for row in rows:
        d = dict(row)
        summary: dict = {}
        if d.get("explanation_summary"):
            try:
                summary = json.loads(d["explanation_summary"])
            except Exception:
                pass
        result.append({
            "claim_id": d["claim_id"],
            "claim_text": d["claim_text"],
            "timestamp": d["timestamp_start"],
            "verdict_label": d.get("verdict_label") or "UNVERIFIABLE",
            "composite_confidence_score": d.get("composite_confidence_score") or 0.0,
            "explanation": summary.get("explanation", ""),
            "sources": summary.get("sources", []),
        })
    return result


async def get_pending_count(session_uuid: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT COUNT(*) FROM claims WHERE session_uuid = ? AND verification_status NOT IN ('VERIFIED', 'FAILED')",
            (session_uuid,),
        ) as cursor:
            row = await cursor.fetchone()
    return row[0] if row else 0


async def get_session_status(session_uuid: str) -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM claims WHERE session_uuid = ?",
            (session_uuid,),
        ) as cursor:
            rows = await cursor.fetchall()
            claims = [dict(row) for row in rows]

        async with db.execute(
            "SELECT timeline_json FROM sessions WHERE session_uuid = ?",
            (session_uuid,),
        ) as cursor:
            session_row = await cursor.fetchone()
            timeline = session_row["timeline_json"] if session_row else None

    return {"claims": claims, "timeline": timeline}
