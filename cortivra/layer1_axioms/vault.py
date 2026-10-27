"""
Two-tier law registry separating read-only core axioms (`CORE_IMMUTABLE`)
from quarantined runtime discoveries (`DISCOVERED_SANDBOX`).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from cortivra.layer1_axioms.shields import AuditCertificate, ConservationViolationError


@dataclass(frozen=True)
class CrystallizedLawRecord:
    """Signed record of a verified mathematical law."""

    law_id: str
    name: str
    domain: str
    symbolic_expression: str
    state_variables: tuple[dict[str, Any], ...]
    parameters: tuple[dict[str, Any], ...]
    tier: str
    audit_passed: bool
    falsification_score: float
    created_at_utc: str
    sha256_fingerprint: str
    enabled: bool = True
    description: str = ""
    parent_law_ids: tuple[str, ...] = field(default_factory=tuple)
    audit_summary: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @staticmethod
    def compute_fingerprint(
        name: str,
        domain: str,
        symbolic_expression: str,
        state_variables: Sequence[dict[str, Any]],
        parameters: Sequence[dict[str, Any]],
        tier: str,
        parent_law_ids: Sequence[str] = (),
        audit_summary: Sequence[dict[str, Any]] = (),
    ) -> str:
        canonical_payload = {
            "name": name.strip(),
            "domain": domain.strip(),
            "symbolic_expression": " ".join(symbolic_expression.strip().split()),
            "state_variables": list(state_variables),
            "parameters": list(parameters),
            "tier": tier,
            "parent_law_ids": list(parent_law_ids),
            "audit_summary": list(audit_summary),
        }
        serialized = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def verify_integrity(self) -> bool:
        expected = self.compute_fingerprint(
            name=self.name,
            domain=self.domain,
            symbolic_expression=self.symbolic_expression,
            state_variables=self.state_variables,
            parameters=self.parameters,
            tier=self.tier,
            parent_law_ids=self.parent_law_ids,
            audit_summary=self.audit_summary,
        )
        return self.sha256_fingerprint == expected


class LawVault:
    """
    Two-tier law storage.

    Tier 1 (`CORE_IMMUTABLE`): Read-only fundamental laws registered at startup.
    Tier 2 (`DISCOVERED_SANDBOX`): Quarantined storage for runtime-discovered laws.
    """

    TIER_CORE = "CORE_IMMUTABLE"
    TIER_DISCOVERED = "DISCOVERED_SANDBOX"

    def __init__(self, sandbox_storage_path: str | Path | None = None) -> None:
        self._core_laws: dict[str, CrystallizedLawRecord] = {}
        self._discovered_laws: dict[str, CrystallizedLawRecord] = {}
        self._sandbox_path = Path(sandbox_storage_path) if sandbox_storage_path else None

        if self._sandbox_path is not None and self._sandbox_path.exists():
            self._load_sandbox_from_disk()

    def register_core_law(
        self,
        law_id: str,
        name: str,
        domain: str,
        symbolic_expression: str,
        state_variables: Sequence[dict[str, Any]] = (),
        parameters: Sequence[dict[str, Any]] = (),
        description: str = "",
    ) -> CrystallizedLawRecord:
        if law_id in self._core_laws:
            raise PermissionError(
                f"Core law '{law_id}' is locked in Tier 1 (CORE_IMMUTABLE) and cannot be overwritten."
            )
        fp = CrystallizedLawRecord.compute_fingerprint(
            name=name,
            domain=domain,
            symbolic_expression=symbolic_expression,
            state_variables=state_variables,
            parameters=parameters,
            tier=self.TIER_CORE,
            parent_law_ids=(),
        )
        record = CrystallizedLawRecord(
            law_id=law_id,
            name=name,
            domain=domain,
            symbolic_expression=symbolic_expression,
            state_variables=tuple(state_variables),
            parameters=tuple(parameters),
            tier=self.TIER_CORE,
            audit_passed=True,
            falsification_score=1.0,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            sha256_fingerprint=fp,
            enabled=True,
            description=description,
            parent_law_ids=(),
        )
        self._core_laws[law_id] = record
        return record

    def crystallize_discovered_law(
        self,
        law_id: str,
        name: str,
        domain: str,
        symbolic_expression: str,
        audit_certificate: AuditCertificate,
        falsification_score: float,
        state_variables: Sequence[dict[str, Any]] = (),
        parameters: Sequence[dict[str, Any]] = (),
        parent_law_ids: Sequence[str] = (),
        description: str = "",
        min_falsification_score: float = 0.80,
    ) -> CrystallizedLawRecord:
        if law_id in self._core_laws:
            raise PermissionError(
                f"Cannot crystallize discovered law '{law_id}': ID collides with Tier 1 Core Law."
            )
        for pid in parent_law_ids:
            if pid not in self._core_laws and pid not in self._discovered_laws:
                raise KeyError(
                    f"Cannot crystallize '{law_id}': unknown parent_law_id '{pid}' in lineage."
                )

        audit_certificate.raise_if_failed()
        audit_summary = tuple(asdict(r) for r in audit_certificate.results)

        if falsification_score < min_falsification_score:
            raise ConservationViolationError(
                f"Discovered law '{name}' rejected by Vault: falsification_score "
                f"({falsification_score:.3f}) < required minimum ({min_falsification_score:.3f})."
            )

        fp = CrystallizedLawRecord.compute_fingerprint(
            name=name,
            domain=domain,
            symbolic_expression=symbolic_expression,
            state_variables=state_variables,
            parameters=parameters,
            tier=self.TIER_DISCOVERED,
            parent_law_ids=parent_law_ids,
            audit_summary=audit_summary,
        )
        record = CrystallizedLawRecord(
            law_id=law_id,
            name=name,
            domain=domain,
            symbolic_expression=symbolic_expression,
            state_variables=tuple(state_variables),
            parameters=tuple(parameters),
            tier=self.TIER_DISCOVERED,
            audit_passed=True,
            falsification_score=float(falsification_score),
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            sha256_fingerprint=fp,
            enabled=True,
            description=description,
            parent_law_ids=tuple(parent_law_ids),
            audit_summary=audit_summary,
        )
        self._discovered_laws[law_id] = record
        self._persist_sandbox_to_disk()
        return record

    def set_discovered_law_enabled(self, law_id: str, enabled: bool) -> CrystallizedLawRecord:
        if law_id in self._core_laws:
            raise PermissionError("Cannot disable a Tier 1 Core Law.")
        if law_id not in self._discovered_laws:
            raise KeyError(f"Discovered law '{law_id}' not found in Tier 2 sandbox.")
        old = self._discovered_laws[law_id]
        updated = CrystallizedLawRecord(
            law_id=old.law_id,
            name=old.name,
            domain=old.domain,
            symbolic_expression=old.symbolic_expression,
            state_variables=old.state_variables,
            parameters=old.parameters,
            tier=old.tier,
            audit_passed=old.audit_passed,
            falsification_score=old.falsification_score,
            created_at_utc=old.created_at_utc,
            sha256_fingerprint=old.sha256_fingerprint,
            enabled=enabled,
            description=old.description,
            parent_law_ids=old.parent_law_ids,
            audit_summary=old.audit_summary,
        )
        self._discovered_laws[law_id] = updated
        self._persist_sandbox_to_disk()
        return updated

    def clear_discovered_sandbox(self) -> int:
        count = len(self._discovered_laws)
        self._discovered_laws.clear()
        self._persist_sandbox_to_disk()
        return count

    def get_law(self, law_id: str, include_discovered: bool = True) -> CrystallizedLawRecord:
        if law_id in self._core_laws:
            rec = self._core_laws[law_id]
        elif include_discovered and law_id in self._discovered_laws:
            rec = self._discovered_laws[law_id]
            if not rec.enabled:
                raise PermissionError(f"Discovered law '{law_id}' is currently quarantined (disabled).")
        else:
            raise KeyError(f"Law '{law_id}' not found (include_discovered={include_discovered}).")

        if not rec.verify_integrity():
            raise ConservationViolationError(f"Cryptographic integrity check failed for law '{law_id}'.")
        return rec

    def list_laws(
        self,
        domain: str | None = None,
        include_discovered: bool = False,
    ) -> list[CrystallizedLawRecord]:
        records = list(self._core_laws.values())
        if include_discovered:
            records.extend(r for r in self._discovered_laws.values() if r.enabled)
        if domain is not None:
            records = [r for r in records if r.domain == domain]
        return records

    def get_ancestry_chain(self, law_id: str) -> list[CrystallizedLawRecord]:
        """
        Traverse the parent_law_ids DAG of `law_id` in topological order up to its
        root Tier 1 (`CORE_IMMUTABLE`) axioms, verifying cryptographic integrity at every node.
        """
        visited: set[str] = set()
        visiting: set[str] = set()
        ordered: list[CrystallizedLawRecord] = []

        def _dfs(curr_id: str) -> None:
            if curr_id in visited:
                return
            if curr_id in visiting:
                raise ConservationViolationError(f"Cycle detected in law lineage at '{curr_id}'.")
            visiting.add(curr_id)
            rec = self.get_law(curr_id, include_discovered=True)
            for pid in rec.parent_law_ids:
                _dfs(pid)
            visiting.remove(curr_id)
            visited.add(curr_id)
            ordered.append(rec)

        _dfs(law_id)
        return ordered

    def _persist_sandbox_to_disk(self) -> None:
        if self._sandbox_path is None:
            return
        self._sandbox_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {law_id: asdict(rec) for law_id, rec in self._discovered_laws.items()}
        tmp_path = self._sandbox_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_path, self._sandbox_path)

    def _load_sandbox_from_disk(self) -> None:
        if self._sandbox_path is None or not self._sandbox_path.exists():
            return
        raw = json.loads(self._sandbox_path.read_text(encoding="utf-8"))
        for law_id, item in raw.items():
            rec = CrystallizedLawRecord(
                law_id=item["law_id"],
                name=item["name"],
                domain=item["domain"],
                symbolic_expression=item["symbolic_expression"],
                state_variables=tuple(item["state_variables"]),
                parameters=tuple(item["parameters"]),
                tier=item["tier"],
                audit_passed=bool(item["audit_passed"]),
                falsification_score=float(item["falsification_score"]),
                created_at_utc=item["created_at_utc"],
                sha256_fingerprint=item["sha256_fingerprint"],
                enabled=bool(item.get("enabled", True)),
                description=item.get("description", ""),
                parent_law_ids=tuple(item.get("parent_law_ids", ())),
                audit_summary=tuple(item.get("audit_summary", ())),
            )
            if not rec.verify_integrity():
                raise ConservationViolationError(
                    f"Corrupted sandbox vault file: SHA-256 mismatch for '{law_id}'."
                )
            self._discovered_laws[law_id] = rec
