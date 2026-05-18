from __future__ import annotations

import sqlite3

from ppmlx.memory_store import MemoryStore


def _tables(path):
    with sqlite3.connect(path) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row[0] for row in rows}


def test_init_creates_extraction_atom_alias_tables(tmp_path):
    db_path = tmp_path / "memory.db"
    store = MemoryStore(db_path)

    store.init()

    tables = _tables(db_path)
    assert "memory_extraction_jobs" in tables
    assert "memory_atoms" in tables
    assert "memory_entity_aliases" in tables


def test_enqueue_list_claim_complete_and_fail_extraction_jobs(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.init()

    enqueued = store.enqueue_extraction_job(
        {"messages": ["remember this"]},
        job_id="job-test-1",
        source_event_id=None,
        priority=10,
        valid_at="2026-01-01T00:00:00.000",
        metadata={"kind": "unit"},
    )
    assert enqueued["job_id"] == "job-test-1"
    assert enqueued["status"] == "queued"
    assert enqueued["payload"] == {"messages": ["remember this"]}
    assert enqueued["metadata"] == {"kind": "unit"}

    queued = store.list_extraction_jobs(status="queued")
    assert [job["job_id"] for job in queued] == ["job-test-1"]

    claimed = store.claim_extraction_job("worker-a")
    assert claimed is not None
    assert claimed["job_id"] == "job-test-1"
    assert claimed["status"] == "claimed"
    assert claimed["worker_id"] == "worker-a"
    assert claimed["attempts"] == 1
    assert claimed["claimed_at"] is not None
    assert store.claim_extraction_job("worker-b") is None

    assert store.complete_extraction_job("job-test-1", result={"atoms": 2}) is True
    completed = store.get_extraction_job("job-test-1")
    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["result"] == {"atoms": 2}
    assert completed["completed_at"] is not None
    assert completed["invalid_at"] is not None

    store.enqueue_extraction_job({"messages": ["bad"]}, job_id="job-test-2")
    claimed_failed = store.claim_extraction_job("worker-a")
    assert claimed_failed is not None
    assert claimed_failed["job_id"] == "job-test-2"
    assert store.fail_extraction_job("job-test-2", "boom") is True
    failed = store.get_extraction_job("job-test-2")
    assert failed is not None
    assert failed["status"] == "failed"
    assert failed["error"] == "boom"
    assert failed["failed_at"] is not None
    assert failed["invalid_at"] is not None


def test_renew_extraction_job_claim_refreshes_claimed_at(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.init()
    store.enqueue_extraction_job({"messages": ["slow"]}, job_id="job-renew")
    claimed = store.claim_extraction_job("worker-a")
    assert claimed is not None
    with store._connect() as conn:
        conn.execute(
            "UPDATE memory_extraction_jobs SET claimed_at = '2000-01-01T00:00:00.000' WHERE job_id = ?",
            ("job-renew",),
        )
        conn.commit()

    assert store.renew_extraction_job_claim("job-renew", "worker-a") is True

    renewed = store.get_extraction_job("job-renew")
    assert renewed is not None
    assert renewed["status"] == "claimed"
    assert renewed["claimed_at"] != "2000-01-01T00:00:00.000"
    assert store.renew_extraction_job_claim("job-renew", "worker-b") is False


def test_requeue_stale_claimed_extraction_jobs(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.init()
    store.enqueue_extraction_job({"messages": ["slow"]}, job_id="job-stale")
    claimed = store.claim_extraction_job("worker-a")
    assert claimed is not None
    with store._connect() as conn:
        conn.execute(
            "UPDATE memory_extraction_jobs SET claimed_at = '2000-01-01T00:00:00.000' WHERE job_id = ?",
            ("job-stale",),
        )
        conn.commit()

    recovered = store.requeue_stale_claimed_extraction_jobs(stale_after_seconds=1)

    assert recovered == {"requeued": 1, "failed": 0}
    requeued = store.get_extraction_job("job-stale")
    assert requeued is not None
    assert requeued["status"] == "queued"
    assert requeued["worker_id"] is None
    assert requeued["attempts"] == 1
    assert "stale claim requeued" in requeued["error"]
    reclaimed = store.claim_extraction_job("worker-b")
    assert reclaimed is not None
    assert reclaimed["attempts"] == 2


def test_atom_same_slot_without_supersession_does_not_invalidate_prior_atom(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.init()

    store.store_atom(
        {
            "atom_id": "atom-tv-budget-old",
            "type": "constraint",
            "subject": "TV Purchase",
            "predicate": "budget",
            "object": "5000 PLN",
            "scope": "project",
            "valid_at": "2026-01-01T00:00:00.000",
        }
    )
    store.store_atom(
        {
            "atom_id": "atom-tv-budget-new",
            "type": "constraint",
            "subject": "tv purchase",
            "predicate": "budget",
            "object": "6000 PLN",
            "scope": "project",
            "valid_at": "2026-01-02T00:00:00.000",
        }
    )

    atoms = store.query_atoms(type="constraint", predicate="budget", scope="project", active_only=True)
    assert {atom["atom_id"] for atom in atoms} == {"atom-tv-budget-old", "atom-tv-budget-new"}
    assert all(atom["invalid_at"] is None for atom in atoms)
    assert all(atom["expired_at"] is None for atom in atoms)


def test_superseding_atom_closes_prior_conflicting_slot(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.init()

    store.store_atom(
        {
            "atom_id": "atom-tv-budget-old",
            "type": "constraint",
            "subject": "Project TV Purchase",
            "predicate": "budget",
            "object": "5000 PLN",
            "scope": "project",
            "valid_at": "2026-01-01T00:00:00.000",
        }
    )
    current = store.store_atom(
        {
            "atom_id": "atom-tv-budget-new",
            "type": "constraint",
            "subject": "tv purchase",
            "predicate": "budget",
            "object": "6000 PLN",
            "scope": "project",
            "valid_at": "2026-01-02T00:00:00.000",
            "metadata": {"from_now_on": True},
        }
    )

    old = store.get_atom("atom-tv-budget-old")
    assert old is not None
    assert old["invalid_at"] == "2026-01-02T00:00:00.000"
    assert old["expired_at"] == "2026-01-02T00:00:00.000"
    assert current["invalid_at"] is None
    assert current["expired_at"] is None

    active_atoms = store.query_atoms(type="constraint", predicate="budget", scope="project", active_only=True)
    assert [atom["atom_id"] for atom in active_atoms] == ["atom-tv-budget-new"]
    assert active_atoms[0]["object"] == "6000 PLN"


def test_atoms_and_aliases_can_be_stored_and_read(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.init()

    atom = store.store_atom(
        {
            "atom_id": "atom-tv-budget",
            "type": "constraint",
            "subject": "tv purchase",
            "predicate": "budget",
            "object": "5000 PLN",
            "text": "Budget is 5000 PLN",
            "scope": "project",
            "confidence": 0.92,
            "valid_at": "2026-01-01T00:00:00.000",
            "metadata": {"project_id": "tv-shopping"},
        }
    )
    assert atom["atom_id"] == "atom-tv-budget"
    assert atom["metadata"] == {"project_id": "tv-shopping"}

    atoms = store.query_atoms(type="constraint", subject="tv purchase", scope="project")
    assert len(atoms) == 1
    assert atoms[0]["predicate"] == "budget"
    assert atoms[0]["object"] == "5000 PLN"

    alias = store.store_entity_alias(
        {
            "alias_id": "alias-lg-c4",
            "entity_id": "ent_lg_oled_c4",
            "alias": "LG C4",
            "type": "product",
            "scope": "project",
            "confidence": 0.99,
            "metadata": {"brand": "LG"},
        }
    )
    assert alias["alias_id"] == "alias-lg-c4"
    assert alias["metadata"] == {"brand": "LG"}

    aliases = store.query_entity_aliases(alias="LG C4", type="product", scope="project")
    assert len(aliases) == 1
    assert aliases[0]["entity_id"] == "ent_lg_oled_c4"
