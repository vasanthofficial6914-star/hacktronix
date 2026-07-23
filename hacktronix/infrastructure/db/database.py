"""
Database Manager for SQLite Persistence.

Handles connection pooling, table schema creation, index creation, and transaction context.
"""

import os
import sqlite3
from typing import Optional


class DatabaseManager:
    """
    Manages SQLite database connections and initializes full DDL schema.
    """

    def __init__(self, db_path: str = "data/world_model.db") -> None:
        self.db_path = db_path
        db_dir = os.path.dirname(os.path.abspath(db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Returns a new sqlite3 connection with Row factory enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        """Initializes database tables if they do not exist."""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS entities (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            room_id TEXT,
            confidence REAL DEFAULT 1.0,
            states_json TEXT DEFAULT '{}',
            bounding_box_json TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS relationships (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            target_id TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            last_observed REAL NOT NULL,
            FOREIGN KEY(source_id) REFERENCES entities(id) ON DELETE CASCADE,
            FOREIGN KEY(target_id) REFERENCES entities(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS inventory (
            entity_id TEXT PRIMARY KEY,
            acquired_at REAL NOT NULL,
            FOREIGN KEY(entity_id) REFERENCES entities(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS state_history (
            version_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            entity_id TEXT,
            description TEXT NOT NULL,
            snapshot_json TEXT NOT NULL,
            timestamp REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS timeline (
            id TEXT PRIMARY KEY,
            source_type TEXT NOT NULL,
            raw_observation TEXT NOT NULL,
            parsed_json TEXT NOT NULL,
            timestamp REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_entities_category ON entities(category);
        CREATE INDEX IF NOT EXISTS idx_entities_room_id ON entities(room_id);
        CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id);
        CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id);
        """
        with self.get_connection() as conn:
            conn.executescript(schema_sql)
            conn.commit()
