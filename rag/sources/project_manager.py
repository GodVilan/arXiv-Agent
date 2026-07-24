"""
project_manager.py – SQLite-backed workspace manager for Researches, Conversations, and Messages.
"""
import logging
import sqlite3
import json
import uuid
import time
from pathlib import Path
from rag import config

log = logging.getLogger(__name__)


class ProjectManager:
    """
    Manages SQLite database operations for multi-project workspaces (Researches),
    conversations (threads), and full chat message persistence.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or (config.DATA_DIR / "session_papers.db")
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        """Create tables for projects, papers, threads, and logs if not present."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # 1. Researches (projects) table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS researches (
                    research_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    scope_type TEXT NOT NULL, -- 'user_only' or 'all_sources'
                    created_at REAL NOT NULL,
                    is_default INTEGER NOT NULL DEFAULT 0,
                    instructions TEXT DEFAULT '',
                    is_starred INTEGER DEFAULT 0
                )
            """)

            # 2. Papers in Researches mapping table (many-to-many)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS research_papers (
                    research_id TEXT,
                    paper_id TEXT,
                    PRIMARY KEY (research_id, paper_id),
                    FOREIGN KEY (research_id) REFERENCES researches(research_id) ON DELETE CASCADE
                )
            """)

            # 3. Conversations (threads) table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    research_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    use_corpus INTEGER NOT NULL DEFAULT 1,
                    use_arxiv INTEGER NOT NULL DEFAULT 1,
                    use_session INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    is_starred INTEGER NOT NULL DEFAULT 0,
                    mode TEXT NOT NULL DEFAULT 'qa',
                    FOREIGN KEY (research_id) REFERENCES researches(research_id) ON DELETE CASCADE
                )
            """)

            # 4. Messages (logs) table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL, -- 'user' or 'assistant'
                    content TEXT NOT NULL,
                    scratchpad TEXT, -- JSON serialized list of steps
                    created_at REAL NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
                )
            """)

            # 5. Workspace memories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workspace_memories (
                    memory_id TEXT PRIMARY KEY,
                    research_id TEXT NOT NULL,
                    category TEXT NOT NULL, -- 'preference', 'fact', 'hypothesis'
                    content TEXT NOT NULL,
                    importance REAL DEFAULT 0.5,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (research_id) REFERENCES researches(research_id) ON DELETE CASCADE
                )
            """)

            # 6. Workspace entities table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workspace_entities (
                    entity_id TEXT PRIMARY KEY,
                    research_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    FOREIGN KEY (research_id) REFERENCES researches(research_id) ON DELETE CASCADE
                )
            """)

            # 7. Pinned notes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pinned_notes (
                    note_id TEXT PRIMARY KEY,
                    research_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (research_id) REFERENCES researches(research_id) ON DELETE CASCADE
                )
            """)

            conn.commit()
            conn.close()
            log.info("ProjectManager tables initialized successfully.")
        except Exception as e:
            log.error("Failed to initialize database tables: %s", e)

    def _bootstrap_default_research(self) -> None:
        """Create a default 'General Q&A' research workspace if none exists."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT research_id FROM researches WHERE is_default = 1 LIMIT 1")
            row = cursor.fetchone()
            if not row:
                d_id = "default_research"
                cursor.execute("""
                    INSERT INTO researches (research_id, name, scope_type, created_at, is_default, instructions, is_starred)
                    VALUES (?, ?, ?, ?, 1, '', 0)
                """, (d_id, "General Q&A", "all_sources", time.time()))
                conn.commit()
                log.info("Bootstrapped default 'General Q&A' research workspace.")
            conn.close()
        except Exception as e:
            log.error("Failed to bootstrap default research: %s", e)

    # ── Researches CRUD ────────────────────────────────────────────────────────

    def create_research(self, name: str, scope_type: str) -> str:
        """Creates a new Research project workspace and returns its ID."""
        r_id = str(uuid.uuid4())
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO researches (research_id, name, scope_type, created_at, is_default, instructions, is_starred)
                VALUES (?, ?, ?, ?, 0, '', 0)
            """, (r_id, name, scope_type, time.time()))
            conn.commit()
            conn.close()
            log.info("Created research project: %s (ID: %s)", name, r_id)
            return r_id
        except Exception as e:
            log.error("Failed to create research project: %s", e)
            return ""

    def list_researches(self) -> list[dict]:
        """Returns a list of all Research project workspaces, ordered by default first."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT research_id, name, scope_type, created_at, is_default, instructions, is_starred
                FROM researches
                ORDER BY is_default DESC, created_at ASC
            """)
            rows = cursor.fetchall()
            conn.close()
            return [
                {
                    "research_id": r[0],
                    "name": r[1],
                    "scope_type": r[2],
                    "created_at": r[3],
                    "is_default": bool(r[4]),
                    "instructions": r[5] or "",
                    "is_starred": bool(r[6])
                }
                for r in rows
            ]
        except Exception as e:
            log.error("Failed to list researches: %s", e)
            return []

    def get_research(self, research_id: str) -> dict | None:
        """Get a single research workspace by ID."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT research_id, name, scope_type, created_at, is_default, instructions, is_starred
                FROM researches WHERE research_id = ?
            """, (research_id,))
            r = cursor.fetchone()
            conn.close()
            if r:
                return {
                    "research_id": r[0],
                    "name": r[1],
                    "scope_type": r[2],
                    "created_at": r[3],
                    "is_default": bool(r[4]),
                    "instructions": r[5] or "",
                    "is_starred": bool(r[6])
                }
            return None
        except Exception as e:
            log.error("Failed to get research: %s", e)
            return None

    def update_research_instructions(self, research_id: str, instructions: str) -> bool:
        """Update workspace instructions."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE researches SET instructions = ? WHERE research_id = ?
            """, (instructions, research_id))
            conn.commit()
            conn.close()
            log.info("Updated custom instructions for workspace %s", research_id)
            return True
        except Exception as e:
            log.error("Failed to update research instructions: %s", e)
            return False

    def update_research_starred(self, research_id: str, is_starred: bool) -> bool:
        """Toggle starred state of a research project."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE researches SET is_starred = ? WHERE research_id = ?
            """, (int(is_starred), research_id))
            conn.commit()
            conn.close()
            log.info("Updated research starred to %d for %s", int(is_starred), research_id)
            return True
        except Exception as e:
            log.error("Failed to update research starred: %s", e)
            return False

    def update_research_name(self, research_id: str, name: str) -> bool:
        """Rename a research project."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE researches SET name = ? WHERE research_id = ?
            """, (name, research_id))
            conn.commit()
            conn.close()
            log.info("Renamed research project %s to %s", research_id, name)
            return True
        except Exception as e:
            log.error("Failed to rename research project: %s", e)
            return False

    def delete_research(self, research_id: str) -> bool:
        """Deletes a research project workspace (cascade deletes threads & messages)."""
        # Protect default research from deletion
        if research_id == "default_research":
            log.warning("Cannot delete default General Q&A research.")
            return False
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM researches WHERE research_id = ?", (research_id,))
            conn.commit()
            conn.close()
            log.info("Deleted research workspace: %s", research_id)
            return True
        except Exception as e:
            log.error("Failed to delete research: %s", e)
            return False

    # ── Research Papers Mapping ────────────────────────────────────────────────

    def add_paper_to_research(self, research_id: str, paper_id: str) -> bool:
        """Associates an indexed paper ID to a specific Research project workspace."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO research_papers (research_id, paper_id)
                VALUES (?, ?)
            """, (research_id, paper_id))
            conn.commit()
            conn.close()
            log.info("Added paper %s to research workspace %s", paper_id, research_id)
            return True
        except Exception as e:
            log.error("Failed to add paper to research: %s", e)
            return False

    def remove_paper_from_research(self, research_id: str, paper_id: str) -> bool:
        """Removes a paper association from a specific Research project workspace."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM research_papers WHERE research_id = ? AND paper_id = ?
            """, (research_id, paper_id))
            conn.commit()
            conn.close()
            log.info("Removed paper %s from research workspace %s", paper_id, research_id)
            return True
        except Exception as e:
            log.error("Failed to remove paper from research: %s", e)
            return False

    def set_research_papers(self, research_id: str, paper_ids: list[str]) -> bool:
        """Sets the precise list of associated papers for a Research project workspace."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            # Clear existing mappings
            cursor.execute("DELETE FROM research_papers WHERE research_id = ?", (research_id,))
            # Insert new mappings
            for pid in paper_ids:
                cursor.execute("""
                    INSERT INTO research_papers (research_id, paper_id)
                    VALUES (?, ?)
                """, (research_id, pid))
            conn.commit()
            conn.close()
            log.info("Updated research workspace %s with %d papers", research_id, len(paper_ids))
            return True
        except Exception as e:
            log.error("Failed to set research papers: %s", e)
            return False

    def get_research_paper_ids(self, research_id: str) -> set[str]:
        """Returns the set of paper IDs associated with a specific Research workspace."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT paper_id FROM research_papers WHERE research_id = ?", (research_id,))
            rows = cursor.fetchall()
            conn.close()
            return {r[0] for r in rows}
        except Exception as e:
            log.error("Failed to get research paper IDs: %s", e)
            return set()

    # ── Conversations (Threads) CRUD ───────────────────────────────────────────

    def create_conversation(
        self, research_id: str, title: str,
        use_corpus: int = 1, use_arxiv: int = 1, use_session: int = 1
    ) -> str:
        """Creates a new conversation thread inside a Research workspace."""
        c_id = str(uuid.uuid4())
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversations (conversation_id, research_id, title, use_corpus, use_arxiv, use_session, created_at, is_starred, mode)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, 'qa')
            """, (c_id, research_id, title, use_corpus, use_arxiv, use_session, time.time()))
            conn.commit()
            conn.close()
            log.info("Created conversation thread: %s (ID: %s)", title, c_id)
            return c_id
        except Exception as e:
            log.error("Failed to create conversation: %s", e)
            return ""

    def list_conversations(self, research_id: str) -> list[dict]:
        """Returns all conversation threads for a specific Research workspace."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT conversation_id, title, use_corpus, use_arxiv, use_session, created_at, is_starred, mode
                FROM conversations
                WHERE research_id = ?
                ORDER BY created_at DESC
            """, (research_id,))
            rows = cursor.fetchall()
            conn.close()
            return [
                {
                    "conversation_id": r[0],
                    "title": r[1],
                    "use_corpus": bool(r[2]),
                    "use_arxiv": bool(r[3]),
                    "use_session": bool(r[4]),
                    "created_at": r[5],
                    "is_starred": bool(r[6]),
                    "mode": r[7] or "qa"
                }
                for r in rows
            ]
        except Exception as e:
            log.error("Failed to list conversations: %s", e)
            return []

    def get_conversation(self, conversation_id: str) -> dict | None:
        """Get details of a specific conversation thread."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT conversation_id, research_id, title, use_corpus, use_arxiv, use_session, created_at, is_starred, mode
                FROM conversations WHERE conversation_id = ?
            """, (conversation_id,))
            r = cursor.fetchone()
            conn.close()
            if r:
                return {
                    "conversation_id": r[0],
                    "research_id": r[1],
                    "title": r[2],
                    "use_corpus": bool(r[3]),
                    "use_arxiv": bool(r[4]),
                    "use_session": bool(r[5]),
                    "created_at": r[6],
                    "is_starred": bool(r[7]),
                    "mode": r[8] or "qa"
                }
            return None
        except Exception as e:
            log.error("Failed to get conversation: %s", e)
            return None

    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """Update the display title of a conversation thread."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE conversations SET title = ? WHERE conversation_id = ?
            """, (title, conversation_id))
            conn.commit()
            conn.close()
            log.info("Updated conversation title: %s", title)
            return True
        except Exception as e:
            log.error("Failed to update conversation title: %s", e)
            return False

    def update_conversation_starred(self, conversation_id: str, is_starred: bool) -> bool:
        """Toggle starred state of a conversation thread."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE conversations SET is_starred = ? WHERE conversation_id = ?
            """, (int(is_starred), conversation_id))
            conn.commit()
            conn.close()
            log.info("Updated conversation starred to %d for %s", int(is_starred), conversation_id)
            return True
        except Exception as e:
            log.error("Failed to update conversation starred: %s", e)
            return False

    def update_conversation_mode(self, conversation_id: str, mode: str) -> bool:
        """Toggle the mode ('qa' or 'lit_review') of a conversation thread."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE conversations SET mode = ? WHERE conversation_id = ?
            """, (mode, conversation_id))
            conn.commit()
            conn.close()
            log.info("Updated conversation mode to %s for %s", mode, conversation_id)
            return True
        except Exception as e:
            log.error("Failed to update conversation mode: %s", e)
            return False

    def move_conversation_to_project(self, conversation_id: str, research_id: str) -> bool:
        """Associate/move a conversation thread to a different project workspace."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE conversations SET research_id = ? WHERE conversation_id = ?
            """, (research_id, conversation_id))
            conn.commit()
            conn.close()
            log.info("Moved conversation %s to project workspace %s", conversation_id, research_id)
            return True
        except Exception as e:
            log.error("Failed to move conversation to project: %s", e)
            return False

    def update_conversation_toggles(
        self, conversation_id: str, use_corpus: int, use_arxiv: int, use_session: int
    ) -> bool:
        """Update dynamic search source checkboxes for a conversation thread."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE conversations
                SET use_corpus = ?, use_arxiv = ?, use_session = ?
                WHERE conversation_id = ?
            """, (use_corpus, use_arxiv, use_session, conversation_id))
            conn.commit()
            conn.close()
            log.info("Updated conversation toggles for %s", conversation_id)
            return True
        except Exception as e:
            log.error("Failed to update conversation toggles: %s", e)
            return False

    def delete_conversation(self, conversation_id: str) -> bool:
        """Deletes a conversation thread (cascade deletes its messages)."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
            conn.commit()
            conn.close()
            log.info("Deleted conversation: %s", conversation_id)
            return True
        except Exception as e:
            log.error("Failed to delete conversation: %s", e)
            return False

    # ── Messages (Logs) CRUD ───────────────────────────────────────────────────

    def add_message(self, conversation_id: str, role: str, content: str, scratchpad_list: list | None = None) -> str:
        """Saves a user or assistant message, including serialized scratchpads, to SQLite."""
        m_id = str(uuid.uuid4())
        scratchpad_json = json.dumps(scratchpad_list) if scratchpad_list is not None else None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO messages (message_id, conversation_id, role, content, scratchpad, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (m_id, conversation_id, role, content, scratchpad_json, time.time()))
            conn.commit()
            conn.close()
            log.debug("Saved chat message from %s to thread %s", role, conversation_id)
            return m_id
        except Exception as e:
            log.error("Failed to save message to SQLite: %s", e)
            return ""

    def update_message_content(self, message_id: str, content: str) -> bool:
        """Update the content text/payload of an existing message."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE messages SET content = ? WHERE message_id = ?
            """, (content, message_id))
            conn.commit()
            conn.close()
            log.info("Updated content for message %s", message_id)
            return True
        except Exception as e:
            log.error("Failed to update message content: %s", e)
            return False

    def get_messages(self, conversation_id: str) -> list[dict]:
        """Loads and formats all messages for a specific conversation thread."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT message_id, role, content, scratchpad, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
            """, (conversation_id,))
            rows = cursor.fetchall()
            conn.close()
            return [
                {
                    "message_id": r[0],
                    "role": r[1],
                    "content": r[2],
                    "scratchpad": json.loads(r[3]) if r[3] else [],
                    "created_at": r[4]
                }
                for r in rows
            ]
        except Exception as e:
            log.error("Failed to get messages: %s", e)
            return []

    # ── Workspace Memories & Entities CRUD ─────────────────────────────────────

    def add_workspace_memory(self, research_id: str, category: str, content: str, importance: float = 0.5) -> str:
        """Saves a workspace memory fact/preference/hypothesis to SQLite."""
        m_id = str(uuid.uuid4())
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO workspace_memories (memory_id, research_id, category, content, importance, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (m_id, research_id, category, content, importance, time.time()))
            conn.commit()
            conn.close()
            log.info("Saved workspace memory of category %s for research %s", category, research_id)
            return m_id
        except Exception as e:
            log.error("Failed to save workspace memory: %s", e)
            return ""

    def get_workspace_memories(self, research_id: str) -> list[dict]:
        """Loads all memories for a workspace research project."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT memory_id, category, content, importance, created_at
                FROM workspace_memories
                WHERE research_id = ?
                ORDER BY created_at DESC
            """, (research_id,))
            rows = cursor.fetchall()
            conn.close()
            return [
                {
                    "memory_id": r[0],
                    "category": r[1],
                    "content": r[2],
                    "importance": r[3],
                    "created_at": r[4]
                }
                for r in rows
            ]
        except Exception as e:
            log.error("Failed to load workspace memories: %s", e)
            return []

    def delete_workspace_memory(self, memory_id: str) -> bool:
        """Deletes a specific workspace memory by ID."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM workspace_memories WHERE memory_id = ?", (memory_id,))
            conn.commit()
            conn.close()
            log.info("Deleted workspace memory: %s", memory_id)
            return True
        except Exception as e:
            log.error("Failed to delete workspace memory: %s", e)
            return False

    def sync_extracted_entities(self, research_id: str, entities_list: list[dict]) -> bool:
        """Syncs the extracted scientific entities to SQLite for a workspace."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            # Clear existing workspace entities
            cursor.execute("DELETE FROM workspace_entities WHERE research_id = ?", (research_id,))
            # Insert newly synced entities
            for ent in entities_list:
                e_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO workspace_entities (entity_id, research_id, name, description, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (e_id, research_id, ent.get("name", "").strip(), ent.get("description", "").strip(), time.time()))
            conn.commit()
            conn.close()
            log.info("Synced %d entities for workspace %s", len(entities_list), research_id)
            return True
        except Exception as e:
            log.error("Failed to sync workspace entities: %s", e)
            return False

    def get_workspace_entities(self, research_id: str) -> list[dict]:
        """Loads all entities registered to a workspace research project."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT entity_id, name, description, created_at
                FROM workspace_entities
                WHERE research_id = ?
                ORDER BY name ASC
            """, (research_id,))
            rows = cursor.fetchall()
            conn.close()
            return [
                {
                    "entity_id": r[0],
                    "name": r[1],
                    "description": r[2],
                    "created_at": r[3]
                }
                for r in rows
            ]
        except Exception as e:
            log.error("Failed to load workspace entities: %s", e)
            return []

    # ── Workspace Pinned Notes CRUD ──────────────────────────────────────────
    def add_pinned_note(self, research_id: str, content: str) -> str:
        """Saves a pinned note/message to SQLite for the research workspace."""
        note_id = str(uuid.uuid4())
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO pinned_notes (note_id, research_id, content, created_at)
                VALUES (?, ?, ?, ?)
            """, (note_id, research_id, content, time.time()))
            conn.commit()
            conn.close()
            log.info("Saved pinned note %s for workspace %s", note_id, research_id)
            return note_id
        except Exception as e:
            log.error("Failed to save pinned note: %s", e)
            return ""

    def get_pinned_notes(self, research_id: str) -> list[dict]:
        """Loads all pinned notes for a research workspace."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT note_id, content, created_at
                FROM pinned_notes
                WHERE research_id = ?
                ORDER BY created_at DESC
            """, (research_id,))
            rows = cursor.fetchall()
            conn.close()
            return [
                {
                    "note_id": r[0],
                    "content": r[1],
                    "created_at": r[2]
                }
                for r in rows
            ]
        except Exception as e:
            log.error("Failed to load pinned notes: %s", e)
            return []

    def delete_pinned_note(self, note_id: str) -> bool:
        """Deletes a specific pinned note by ID."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM pinned_notes WHERE note_id = ?", (note_id,))
            conn.commit()
            conn.close()
            log.info("Deleted pinned note %s", note_id)
            return True
        except Exception as e:
            log.error("Failed to delete pinned note: %s", e)
            return False

    # ── Global Message Search ────────────────────────────────────────────────
    def search_messages(self, query: str, research_id: str) -> list[dict]:
        """Search across all messages in a specific project workspace."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT m.message_id, m.conversation_id, c.title, m.content, m.created_at
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.conversation_id
                WHERE c.research_id = ? AND m.content LIKE ?
                ORDER BY m.created_at DESC
            """, (research_id, f"%{query}%"))
            rows = cursor.fetchall()
            conn.close()
            return [
                {
                    "message_id": r[0],
                    "conversation_id": r[1],
                    "conversation_title": r[2],
                    "content": r[3],
                    "created_at": r[4]
                }
                for r in rows
            ]
        except Exception as e:
            log.error("Failed to search messages: %s", e)
            return []

    # ── Conversation Branching ───────────────────────────────────────────────
    def branch_conversation(self, conversation_id: str, up_to_message_id: str) -> str:
        """Branches a conversation thread up to a specific message ID, creating a pre-populated thread."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 1. Fetch original conversation metadata
            cursor.execute("""
                SELECT research_id, title, use_corpus, use_arxiv, use_session, mode
                FROM conversations
                WHERE conversation_id = ?
            """, (conversation_id,))
            conv = cursor.fetchone()
            if not conv:
                conn.close()
                return ""
            
            research_id, original_title, use_corpus, use_arxiv, use_session, mode = conv
            
            # 2. Get target message timestamp/creation threshold
            cursor.execute("SELECT created_at FROM messages WHERE message_id = ?", (up_to_message_id,))
            msg_row = cursor.fetchone()
            if not msg_row:
                conn.close()
                return ""
            target_time = msg_row[0]
            
            # 3. Create new conversation branched thread
            new_cid = str(uuid.uuid4())
            new_title = f"🔀 Branch of {original_title[:20]}"
            cursor.execute("""
                INSERT INTO conversations (conversation_id, research_id, title, use_corpus, use_arxiv, use_session, created_at, is_starred, mode)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
            """, (new_cid, research_id, new_title, use_corpus, use_arxiv, use_session, time.time(), mode))
            
            # 4. Fetch all messages up to that timestamp
            cursor.execute("""
                SELECT role, content, scratchpad, created_at
                FROM messages
                WHERE conversation_id = ? AND created_at <= ?
                ORDER BY created_at ASC
            """, (conversation_id, target_time))
            messages_to_copy = cursor.fetchall()
            
            # 5. Insert messages into new branched conversation
            for msg in messages_to_copy:
                new_mid = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO messages (message_id, conversation_id, role, content, scratchpad, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (new_mid, new_cid, msg[0], msg[1], msg[2], msg[3]))
                
            conn.commit()
            conn.close()
            log.info("Branched conversation %s up to %s into new thread %s", conversation_id, up_to_message_id, new_cid)
            return new_cid
        except Exception as e:
            log.error("Failed to branch conversation: %s", e)
            return ""
