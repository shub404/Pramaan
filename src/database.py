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
        await db.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                claim_id                  TEXT PRIMARY KEY,
                session_uuid              TEXT NOT NULL REFERENCES sessions(session_uuid),
                timestamp_start           INTEGER,
                timestamp_end             INTEGER,
                claim_text                TEXT,
                importance_score          INTEGER,
                verification_status       TEXT,
                composite_confidence_score REAL,
                verdict_label             TEXT,
                explanation_summary       TEXT
            )
        """)
        await db.commit()


async def create_session(session_uuid: str, video_url: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO sessions (session_uuid, video_url, created_at) VALUES (?, ?, ?)",
            (session_uuid, video_url, datetime.now(timezone.utc).isoformat()),
        )
        await db.commit()


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
                verification_status        = 'VERIFIED'
            WHERE claim_id = ?
            """,
            (confidence, label, summary, claim_id),
        )
        await db.commit()


async def get_session_status(session_uuid: str) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM claims WHERE session_uuid = ?",
            (session_uuid,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
