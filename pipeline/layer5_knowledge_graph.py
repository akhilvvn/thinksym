

import json
import importlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

try:
    _neo4j_mod = importlib.import_module("neo4j")
    _neo4j_exc_mod = importlib.import_module("neo4j.exceptions")
    GraphDatabase = _neo4j_mod.GraphDatabase
    ServiceUnavailable = _neo4j_exc_mod.ServiceUnavailable
    AuthError = _neo4j_exc_mod.AuthError
    _NEO4J_DRIVER_AVAILABLE = True
except ImportError:
    GraphDatabase = None
    ServiceUnavailable = Exception
    AuthError = Exception
    _NEO4J_DRIVER_AVAILABLE = False

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


_neo4j_cfg: dict = {"uri": None, "user": "neo4j", "password": None, "database": "neo4j"}
_neo4j_driver  = None
_neo4j_ok: bool = False


def configure_neo4j(
    uri:      str,
    user:     str,
    password: str,
    database: str = "neo4j",
) -> bool:

    global _neo4j_cfg, _neo4j_driver, _neo4j_ok, _kg_instance
    _neo4j_cfg = {"uri": uri, "user": user, "password": password, "database": database}
    _neo4j_driver = None
    _neo4j_ok = False
    _kg_instance = None
    ok = _get_driver() is not None
    return ok


def _get_driver():

    global _neo4j_driver, _neo4j_ok
    if not _NEO4J_DRIVER_AVAILABLE:
        return None
    if not _neo4j_cfg["uri"] or not _neo4j_cfg["password"]:
        return None
    if _neo4j_driver is not None and _neo4j_ok:
        return _neo4j_driver
    try:
        drv = GraphDatabase.driver(
            _neo4j_cfg["uri"],
            auth=(_neo4j_cfg["user"], _neo4j_cfg["password"]),
            connection_timeout=5,
            max_connection_lifetime=300,
        )
        drv.verify_connectivity()
        _neo4j_driver = drv
        _neo4j_ok = True
        logger.info("Neo4j connected at %s (db=%s)", _neo4j_cfg["uri"], _neo4j_cfg["database"])
        return drv
    except (ServiceUnavailable, AuthError, Exception) as exc:
        logger.warning("Neo4j unavailable (%s) — using NetworkX fallback", exc)
        _neo4j_driver = None
        _neo4j_ok = False
        return None


@dataclass
class AttackMatch:

    node_id:        str
    node_type:      str
    name:           str
    threat_key:     str
    confidence:     float
    owasp_id:       str
    mitre_id:       str
    severity:       str
    description:    str
    matched_via:    list[str] = field(default_factory=list)
    related_nodes:  list[str] = field(default_factory=list)


@dataclass
class GraphStats:

    total_nodes:      int
    total_edges:      int
    node_type_counts: dict[str, int]
    edge_type_counts: dict[str, int]


@dataclass
class KnowledgeGraphResult:


    threat_signals:  dict[str, float]


    matched_attacks: list[AttackMatch] = field(default_factory=list)
    top_attack:      AttackMatch | None = None


    total_matches:   int   = 0
    owasp_hits:      list[str] = field(default_factory=list)
    mitre_hits:      list[str] = field(default_factory=list)
    kg_risk_score:   float = 0.0
    attack_vector_summary: str = ""


    graph_stats:     GraphStats | None = None
    processing_time_ms: float = 0.0

    @property
    def risk_label(self) -> str:
        if self.kg_risk_score >= 0.80: return "CRITICAL"
        if self.kg_risk_score >= 0.60: return "HIGH"
        if self.kg_risk_score >= 0.40: return "MEDIUM"
        if self.kg_risk_score >= 0.20: return "LOW"
        return "SAFE"


class NetworkXBackend:


    def __init__(self):
        self.G = nx.DiGraph()
        self._built = False

    def build(self) -> GraphStats:

        if self._built:
            return self._get_stats()

        logger.info("Building security knowledge graph …")
        t0 = time.perf_counter()


        with open(DATA_DIR / "owasp_llm_top10.json",  encoding="utf-8") as f:
            owasp_data = json.load(f)
        with open(DATA_DIR / "mitre_attack_llm.json", encoding="utf-8") as f:
            mitre_data = json.load(f)

        self._build_owasp_subgraph(owasp_data)
        self._build_mitre_subgraph(mitre_data)
        self._add_harmbench_signatures()
        self._add_cross_edges()

        self._built = True
        stats = self._get_stats()
        logger.info(
            "Knowledge graph built: %d nodes, %d edges in %.0f ms",
            stats.total_nodes, stats.total_edges,
            (time.perf_counter() - t0) * 1000,
        )
        return stats

    def _build_owasp_subgraph(self, data: dict):


        self.G.add_node("OWASP_ROOT", type="ROOT", name="OWASP LLM Top 10 2025")

        for cat in data["categories"]:
            owasp_id = cat["id"]
            node_id  = f"OWASP_{owasp_id}"


            self.G.add_node(
                node_id,
                type        = "OWASP",
                owasp_id    = owasp_id,
                name        = cat["name"],
                threat_key  = cat["threat_key"],
                description = cat["description"],
                severity    = cat["severity"],
                cwe         = cat.get("cwe", ""),
                mitre_id    = cat.get("mitre_technique", ""),
            )
            self.G.add_edge("OWASP_ROOT", node_id, relation="CONTAINS")


            for kw in cat.get("keywords", []):
                kw_node_id = f"KW_{hash(kw)}"
                if not self.G.has_node(kw_node_id):
                    self.G.add_node(
                        kw_node_id,
                        type     = "INDICATOR",
                        subtype  = "KEYWORD",
                        value    = kw,
                        name     = f"Keyword: {kw[:40]}",
                    )
                self.G.add_edge(kw_node_id, node_id, relation="INDICATES", weight=0.7)

    def _build_mitre_subgraph(self, data: dict):

        self.G.add_node("MITRE_ROOT", type="ROOT", name="MITRE ATT&CK for AI")

        for tactic in data.get("tactics", []):
            tactic_id   = tactic["tactic_id"]
            tactic_node = f"TACTIC_{tactic_id}"
            self.G.add_node(
                tactic_node,
                type  = "TACTIC",
                id    = tactic_id,
                name  = tactic["tactic_name"],
            )
            self.G.add_edge("MITRE_ROOT", tactic_node, relation="CONTAINS")

            for tech in tactic.get("techniques", []):
                tech_id   = tech["technique_id"]
                tech_node = f"TECH_{tech_id}"
                self.G.add_node(
                    tech_node,
                    type        = "TECHNIQUE",
                    mitre_id    = tech_id,
                    name        = tech["name"],
                    threat_key  = tech["threat_key"],
                    description = tech["description"],
                    severity    = "HIGH",
                    owasp_id    = "",
                )
                self.G.add_edge(tactic_node, tech_node, relation="USES")


                for indicator in tech.get("indicators", []):
                    ind_node = f"IND_{hash(indicator)}"
                    if not self.G.has_node(ind_node):
                        self.G.add_node(
                            ind_node,
                            type    = "INDICATOR",
                            subtype = "PHRASE",
                            value   = indicator,
                            name    = f"Indicator: {indicator[:40]}",
                        )
                    self.G.add_edge(ind_node, tech_node, relation="INDICATES", weight=0.8)


                for sub in tech.get("subtechniques", []):
                    sub_node = f"SUBTECH_{tech_id}_{hash(sub)}"
                    self.G.add_node(
                        sub_node,
                        type  = "SUBTECHNIQUE",
                        name  = sub,
                        parent = tech_id,
                    )
                    self.G.add_edge(tech_node, sub_node, relation="HAS_SUBTECHNIQUE")

    def _add_harmbench_signatures(self):

        harmbench_sigs = [
            {"id": "HB-001", "name": "DAN Jailbreak Pattern",       "pattern": "DAN mode",         "threat_key": "jailbreak",         "confidence": 0.92},
            {"id": "HB-002", "name": "Developer Mode Bypass",        "pattern": "developer mode",   "threat_key": "jailbreak",         "confidence": 0.89},
            {"id": "HB-003", "name": "Base64 Injection",             "pattern": "base64",           "threat_key": "prompt_injection",  "confidence": 0.78},
            {"id": "HB-004", "name": "Roleplay Persona Bypass",      "pattern": "act as",           "threat_key": "jailbreak",         "confidence": 0.70},
            {"id": "HB-005", "name": "System Prompt Repetition",     "pattern": "repeat the words", "threat_key": "system_prompt_leak","confidence": 0.88},
            {"id": "HB-006", "name": "CBRN Content Request",         "pattern": "nerve agent",      "threat_key": "harmful_content",   "confidence": 0.97},
            {"id": "HB-007", "name": "Fake Authority Injection",     "pattern": "[[SYSTEM",         "threat_key": "prompt_injection",  "confidence": 0.91},
            {"id": "HB-008", "name": "Multi-turn Escalation Pattern","pattern": "gradual",          "threat_key": "jailbreak",         "confidence": 0.85},
            {"id": "HB-009", "name": "Hypothetical Framing Bypass",  "pattern": "hypothetically",   "threat_key": "harmful_content",   "confidence": 0.74},
            {"id": "HB-010", "name": "Encoding Attack",              "pattern": "translate this",   "threat_key": "prompt_injection",  "confidence": 0.76},
            {"id": "HB-011", "name": "Reverse Shell Generation",     "pattern": "reverse shell",    "threat_key": "code_exploit",      "confidence": 0.95},
            {"id": "HB-012", "name": "Phishing Generation",          "pattern": "phishing email",   "threat_key": "social_engineering","confidence": 0.86},
            {"id": "HB-013", "name": "Token Flooding Attack",        "pattern": "repeat this",      "threat_key": "resource_abuse",    "confidence": 0.80},
            {"id": "HB-014", "name": "Safe Cracking Request",        "pattern": "bank safe",        "threat_key": "harmful_content",   "confidence": 0.94},
            {"id": "HB-015", "name": "Website Hacking Request",      "pattern": "hack website",     "threat_key": "code_exploit",      "confidence": 0.95},
            {"id": "HB-016", "name": "Lock Bypass Burglary",         "pattern": "bypass a lock",    "threat_key": "harmful_content",   "confidence": 0.91},
            {"id": "HB-017", "name": "Wire Transfer Fraud",          "pattern": "wire transfer",    "threat_key": "social_engineering","confidence": 0.89},
            {"id": "HB-018", "name": "Credential Dumping",           "pattern": "dump credentials", "threat_key": "code_exploit",      "confidence": 0.93},
        ]
        self.G.add_node("HARMBENCH_ROOT", type="ROOT", name="HarmBench Attack Signatures")
        for sig in harmbench_sigs:
            node_id = f"HB_{sig['id']}"
            self.G.add_node(
                node_id,
                type        = "HARMBENCH",
                sig_id      = sig["id"],
                name        = sig["name"],
                pattern     = sig["pattern"],
                threat_key  = sig["threat_key"],
                confidence  = sig["confidence"],
                severity    = "HIGH" if sig["confidence"] >= 0.85 else "MEDIUM",
                description = f"HarmBench attack signature: {sig['name']}",
                owasp_id    = "",
                mitre_id    = "",
            )
            self.G.add_edge("HARMBENCH_ROOT", node_id, relation="CONTAINS")


            ind_node = f"IND_{hash(sig['pattern'])}"
            if not self.G.has_node(ind_node):
                self.G.add_node(
                    ind_node,
                    type    = "INDICATOR",
                    subtype = "SIGNATURE",
                    value   = sig["pattern"],
                    name    = f"Sig: {sig['pattern'][:40]}",
                )
            self.G.add_edge(ind_node, node_id, relation="MATCHES", weight=sig["confidence"])

    def _add_cross_edges(self):

        cross_maps = [
            ("OWASP_LLM01", "TECH_AML.T0051",  "MAPS_TO"),
            ("OWASP_LLM01", "TECH_AML.T0048",  "MAPS_TO"),
            ("OWASP_LLM01", "TECH_AML.T0049",  "MAPS_TO"),
            ("OWASP_LLM02", "TECH_AML.T0052",  "MAPS_TO"),
            ("OWASP_LLM07", "TECH_AML.T0061",  "MAPS_TO"),
            ("OWASP_LLM05", "TECH_AML.T0049",  "MAPS_TO"),
            ("OWASP_LLM09", "TECH_AML.T0054",  "MAPS_TO"),
            ("OWASP_LLM10", "TECH_AML.T0057",  "MAPS_TO"),

            ("OWASP_LLM01", "OWASP_LLM05",     "ESCALATES_TO"),
            ("OWASP_LLM07", "OWASP_LLM02",     "ESCALATES_TO"),
            ("OWASP_LLM09", "OWASP_LLM05",     "ESCALATES_TO"),
        ]
        for src, dst, rel in cross_maps:
            if self.G.has_node(src) and self.G.has_node(dst):
                self.G.add_edge(src, dst, relation=rel)

    def _get_stats(self) -> GraphStats:
        node_type_counts: dict[str, int] = {}
        edge_type_counts: dict[str, int] = {}
        for _, data in self.G.nodes(data=True):
            t = data.get("type", "UNKNOWN")
            node_type_counts[t] = node_type_counts.get(t, 0) + 1
        for _, _, data in self.G.edges(data=True):
            r = data.get("relation", "UNKNOWN")
            edge_type_counts[r] = edge_type_counts.get(r, 0) + 1
        return GraphStats(
            total_nodes      = self.G.number_of_nodes(),
            total_edges      = self.G.number_of_edges(),
            node_type_counts = node_type_counts,
            edge_type_counts = edge_type_counts,
        )


    def get_stats(self) -> GraphStats:
        return self._get_stats()

    def match_threats(
        self,
        prompt: str,
        threat_signals: dict[str, float],
        top_k: int = 8,
    ) -> list[AttackMatch]:

        prompt_lower = prompt.lower()
        matches: dict[str, AttackMatch] = {}


        matched_indicators: list[tuple[str, float]] = []
        for node, data in self.G.nodes(data=True):
            if data.get("type") == "INDICATOR":
                indicator_val = data.get("value", "").lower()
                if indicator_val and indicator_val in prompt_lower:
                    matched_indicators.append((node, 1.0))


        for ind_node, ind_conf in matched_indicators:
            ind_value = self.G.nodes[ind_node].get("value", "")
            for _, target, edge_data in self.G.out_edges(ind_node, data=True):
                target_data = self.G.nodes[target]
                target_type = target_data.get("type", "")

                if target_type not in ("OWASP", "TECHNIQUE", "HARMBENCH"):
                    continue

                edge_weight  = edge_data.get("weight", 0.5)
                threat_key   = target_data.get("threat_key", "")
                neural_boost = threat_signals.get(threat_key, 0.0)

                # Lower threshold for neural corroboration to 0.05
                # High-confidence indicators (0.85+) count even with minimal neural signal
                if neural_boost < 0.05 and ind_conf <= 1.0:
                    # If no neural signal, still allow high-confidence indicators
                    if edge_weight >= 0.80:  # High-confidence indicators like HB sigs
                        confidence = round(0.50 * edge_weight + 0.15, 4)
                    else:
                        # Discount low-confidence keyword-only matches
                        confidence = round(0.25 * ind_conf + 0.15 * edge_weight, 4)
                else:
                    confidence = round(
                        0.35 * ind_conf +
                        0.35 * edge_weight +
                        0.30 * neural_boost,
                        4
                    )

                if target not in matches or matches[target].confidence < confidence:

                    related = [
                        self.G.nodes[n].get("name", n)
                        for n in nx.neighbors(self.G, target)
                        if n not in ("OWASP_ROOT", "MITRE_ROOT", "HARMBENCH_ROOT")
                    ][:5]


                    owasp_id = target_data.get("owasp_id", "")
                    if not owasp_id and target.startswith("OWASP_"):
                        owasp_id = target.replace("OWASP_", "")

                    matches[target] = AttackMatch(
                        node_id     = target,
                        node_type   = target_type,
                        name        = target_data.get("name", target),
                        threat_key  = threat_key,
                        confidence  = confidence,
                        owasp_id    = owasp_id,
                        mitre_id    = target_data.get("mitre_id", ""),
                        severity    = target_data.get("severity", "MEDIUM"),
                        description = target_data.get("description", ""),
                        matched_via = [ind_value],
                        related_nodes = related,
                    )
                else:
                    matches[target].matched_via.append(ind_value)
                    matches[target].confidence = min(
                        matches[target].confidence + 0.05, 1.0
                    )


        for cat, score in threat_signals.items():
            if score >= 0.25:

                for node, data in self.G.nodes(data=True):
                    if data.get("type") == "OWASP" and data.get("threat_key") == cat:
                        if node not in matches:
                            matches[node] = AttackMatch(
                                node_id     = node,
                                node_type   = "OWASP",
                                name        = data.get("name", node),
                                threat_key  = cat,
                                confidence  = round(score * 0.90, 4),
                                owasp_id    = data.get("owasp_id", ""),
                                mitre_id    = data.get("mitre_id", ""),
                                severity    = data.get("severity", "MEDIUM"),
                                description = data.get("description", ""),
                                matched_via = [f"neural_signal:{score:.3f}"],
                            )
                        break


        sorted_matches = sorted(matches.values(), key=lambda m: m.confidence, reverse=True)
        return sorted_matches[:top_k]


_HARMBENCH_SIGS = [
    {"id": "HB-001", "name": "DAN Jailbreak Pattern",        "pattern": "DAN mode",         "threat_key": "jailbreak",          "confidence": 0.92},
    {"id": "HB-002", "name": "Developer Mode Bypass",         "pattern": "developer mode",   "threat_key": "jailbreak",          "confidence": 0.89},
    {"id": "HB-003", "name": "Base64 Injection",              "pattern": "base64",           "threat_key": "prompt_injection",   "confidence": 0.78},
    {"id": "HB-004", "name": "Roleplay Persona Bypass",       "pattern": "act as",           "threat_key": "jailbreak",          "confidence": 0.70},
    {"id": "HB-005", "name": "System Prompt Repetition",      "pattern": "repeat the words", "threat_key": "system_prompt_leak", "confidence": 0.88},
    {"id": "HB-006", "name": "CBRN Content Request",          "pattern": "nerve agent",      "threat_key": "harmful_content",    "confidence": 0.97},
    {"id": "HB-007", "name": "Fake Authority Injection",      "pattern": "[[system",         "threat_key": "prompt_injection",   "confidence": 0.91},
    {"id": "HB-008", "name": "Multi-turn Escalation Pattern", "pattern": "gradually",        "threat_key": "jailbreak",          "confidence": 0.85},
    {"id": "HB-009", "name": "Hypothetical Framing Bypass",   "pattern": "hypothetically",   "threat_key": "harmful_content",    "confidence": 0.74},
    {"id": "HB-010", "name": "Encoding Attack",               "pattern": "translate this",   "threat_key": "prompt_injection",   "confidence": 0.76},
    {"id": "HB-011", "name": "Reverse Shell Generation",      "pattern": "reverse shell",    "threat_key": "code_exploit",       "confidence": 0.95},
    {"id": "HB-012", "name": "Phishing Generation",           "pattern": "phishing email",   "threat_key": "social_engineering", "confidence": 0.86},
    {"id": "HB-013", "name": "Token Flooding Attack",         "pattern": "repeat this",      "threat_key": "resource_abuse",     "confidence": 0.80},
    {"id": "HB-014", "name": "Safe Cracking Request",         "pattern": "bank safe",        "threat_key": "harmful_content",    "confidence": 0.94},
    {"id": "HB-015", "name": "Website Hacking Request",       "pattern": "hack website",     "threat_key": "code_exploit",       "confidence": 0.95},
    {"id": "HB-016", "name": "Lock Bypass Burglary",          "pattern": "bypass a lock",    "threat_key": "harmful_content",    "confidence": 0.91},
    {"id": "HB-017", "name": "Wire Transfer Fraud",           "pattern": "wire transfer",    "threat_key": "social_engineering", "confidence": 0.89},
    {"id": "HB-018", "name": "Credential Dumping",            "pattern": "dump credentials", "threat_key": "code_exploit",       "confidence": 0.93},
]


class Neo4jBackend:


    def __init__(self, driver, database: str = "neo4j"):
        self.driver   = driver
        self.database = database
        self._built   = False


    def build(self, owasp_data: dict, mitre_data: dict) -> GraphStats:
        with self.driver.session(database=self.database) as s:
            existing = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            if existing > 20:
                logger.info("Neo4j graph already populated (%d nodes) — skipping rebuild", existing)
                self._built = True
                return self.get_stats()

        logger.info("Building knowledge graph in Neo4j (db=%s) …", self.database)
        t0 = time.perf_counter()

        with self.driver.session(database=self.database) as s:

            for stmt in [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (n:OWASPCategory)   REQUIRE n.owasp_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Indicator)        REQUIRE n.value    IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (n:MITRETechnique)   REQUIRE n.mitre_id IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (n:HarmBenchSig)     REQUIRE n.sig_id   IS UNIQUE",
            ]:
                s.run(stmt)


            for cat in owasp_data["categories"]:
                s.run("""
                    MERGE (n:OWASPCategory {owasp_id: $oid})
                    SET n.name=$name, n.threat_key=$tk, n.severity=$sev,
                        n.description=$desc, n.mitre_id=$mid
                """, oid=cat["id"], name=cat["name"], tk=cat["threat_key"],
                    sev=cat["severity"], desc=cat["description"],
                    mid=cat.get("mitre_technique", ""))

                for kw in cat.get("keywords", []):
                    s.run("""
                        MERGE (ind:Indicator {value: $val})
                          SET ind.subtype='KEYWORD', ind.name=$iname
                        WITH ind
                        MATCH (n:OWASPCategory {owasp_id: $oid})
                        MERGE (ind)-[:INDICATES {weight: 0.7}]->(n)
                    """, val=kw.lower(), iname=f"Keyword: {kw[:40]}", oid=cat["id"])


            for tactic in mitre_data.get("tactics", []):
                s.run("""
                    MERGE (t:MITRETactic {tactic_id: $tid})
                      SET t.name=$name
                """, tid=tactic["tactic_id"], name=tactic["tactic_name"])

                for tech in tactic.get("techniques", []):
                    s.run("""
                        MERGE (tech:MITRETechnique {mitre_id: $mid})
                          SET tech.name=$name, tech.threat_key=$tk,
                              tech.description=$desc, tech.severity='HIGH'
                        WITH tech
                        MATCH (t:MITRETactic {tactic_id: $tid})
                        MERGE (t)-[:USES]->(tech)
                    """, mid=tech["technique_id"], name=tech["name"],
                        tk=tech["threat_key"], desc=tech["description"],
                        tid=tactic["tactic_id"])

                    for ind in tech.get("indicators", []):
                        s.run("""
                            MERGE (ind:Indicator {value: $val})
                              SET ind.subtype='PHRASE', ind.name=$iname
                            WITH ind
                            MATCH (tech:MITRETechnique {mitre_id: $mid})
                            MERGE (ind)-[:INDICATES {weight: 0.8}]->(tech)
                        """, val=ind.lower(), iname=f"Indicator: {ind[:40]}",
                            mid=tech["technique_id"])


            for sig in _HARMBENCH_SIGS:
                s.run("""
                    MERGE (hb:HarmBenchSig {sig_id: $sid})
                      SET hb.name=$name, hb.threat_key=$tk,
                          hb.confidence=$conf,
                          hb.severity=CASE WHEN $conf >= 0.85 THEN 'HIGH' ELSE 'MEDIUM' END,
                          hb.description=$desc, hb.owasp_id='', hb.mitre_id=''
                    WITH hb
                    MERGE (ind:Indicator {value: $pat})
                      SET ind.subtype='SIGNATURE', ind.name=$iname
                    MERGE (ind)-[:MATCHES {weight: $conf}]->(hb)
                """, sid=sig["id"], name=sig["name"], tk=sig["threat_key"],
                    conf=sig["confidence"], desc=f"HarmBench: {sig['name']}",
                    pat=sig["pattern"].lower(), iname=f"Sig: {sig['pattern'][:40]}")

            for oid, mid, rel in [
                ("LLM01", "AML.T0051", "MAPS_TO"),
                ("LLM01", "AML.T0048", "MAPS_TO"),
                ("LLM01", "AML.T0049", "MAPS_TO"),
                ("LLM02", "AML.T0052", "MAPS_TO"),
                ("LLM07", "AML.T0061", "MAPS_TO"),
                ("LLM05", "AML.T0049", "MAPS_TO"),
                ("LLM09", "AML.T0054", "MAPS_TO"),
                ("LLM10", "AML.T0057", "MAPS_TO"),
            ]:
                s.run("""
                    MATCH (o:OWASPCategory {owasp_id: $oid}), (m:MITRETechnique {mitre_id: $mid})
                    MERGE (o)-[r:MAPS_TO]->(m)
                """, oid=oid, mid=mid)

            for src, dst in [
                ("LLM01", "LLM05"),
                ("LLM07", "LLM02"),
                ("LLM09", "LLM05"),
            ]:
                s.run("""
                    MATCH (a:OWASPCategory {owasp_id: $src}), (b:OWASPCategory {owasp_id: $dst})
                    MERGE (a)-[:ESCALATES_TO]->(b)
                """, src=src, dst=dst)

        self._built = True
        stats = self.get_stats()
        logger.info(
            "Neo4j graph built: %d nodes, %d edges in %.0f ms",
            stats.total_nodes, stats.total_edges,
            (time.perf_counter() - t0) * 1000,
        )
        return stats


    def match_threats(
        self,
        prompt: str,
        threat_signals: dict[str, float],
        top_k: int = 8,
    ) -> list[AttackMatch]:
        prompt_lower = prompt.lower()

        with self.driver.session(database=self.database) as s:
            rows = s.run("""
                MATCH (ind:Indicator)-[r:INDICATES|MATCHES]->(target)
                WHERE (target:OWASPCategory OR target:MITRETechnique OR target:HarmBenchSig)
                  AND $prompt CONTAINS ind.value
                RETURN
                    ind.value                             AS indicator,
                    r.weight                              AS weight,
                    target.name                           AS name,
                    labels(target)[0]                     AS target_type,
                    COALESCE(target.threat_key,  '')      AS threat_key,
                    COALESCE(target.owasp_id,    '')      AS owasp_id,
                    COALESCE(target.mitre_id,    '')      AS mitre_id,
                    COALESCE(target.severity, 'MEDIUM')   AS severity,
                    COALESCE(target.description, '')      AS description,
                    elementId(target)                     AS target_eid
            """, prompt=prompt_lower).data()


        bucket: dict[str, AttackMatch] = {}
        for row in rows:
            eid         = row["target_eid"]
            threat_key  = row["threat_key"]
            edge_w      = float(row["weight"] or 0.5)
            neural_b    = threat_signals.get(threat_key, 0.0)
            conf        = round(0.40 + 0.30 * edge_w + 0.30 * neural_b, 4)

            if eid not in bucket:
                bucket[eid] = AttackMatch(
                    node_id     = eid,
                    node_type   = row["target_type"],
                    name        = row["name"],
                    threat_key  = threat_key,
                    confidence  = conf,
                    owasp_id    = row["owasp_id"],
                    mitre_id    = row["mitre_id"],
                    severity    = row["severity"],
                    description = row["description"],
                    matched_via = [row["indicator"]],
                )
            else:
                bucket[eid].matched_via.append(row["indicator"])
                bucket[eid].confidence = min(bucket[eid].confidence + 0.05, 1.0)


        for cat, score in threat_signals.items():
            if score >= 0.40:
                with self.driver.session(database=self.database) as s:
                    row = s.run("""
                        MATCH (n:OWASPCategory {threat_key: $tk})
                        RETURN n.name AS name, n.owasp_id AS oid,
                               COALESCE(n.mitre_id,'') AS mid,
                               n.severity AS sev, n.description AS desc,
                               elementId(n) AS eid
                        LIMIT 1
                    """, tk=cat).single()
                    if row and row["eid"] not in bucket:
                        bucket[row["eid"]] = AttackMatch(
                            node_id     = row["eid"],
                            node_type   = "OWASPCategory",
                            name        = row["name"],
                            threat_key  = cat,
                            confidence  = round(score * 0.85, 4),
                            owasp_id    = row["oid"],
                            mitre_id    = row["mid"],
                            severity    = row["sev"],
                            description = row["desc"],
                            matched_via = [f"neural_signal:{score:.3f}"],
                        )

        return sorted(bucket.values(), key=lambda m: m.confidence, reverse=True)[:top_k]


    def get_stats(self) -> GraphStats:
        with self.driver.session(database=self.database) as s:
            nc = {r["lbl"]: r["cnt"] for r in s.run(
                "MATCH (n) UNWIND labels(n) AS lbl RETURN lbl, count(*) AS cnt").data()}
            ec = {r["rtype"]: r["cnt"] for r in s.run(
                "MATCH ()-[r]->() RETURN type(r) AS rtype, count(*) AS cnt").data()}
            tn = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            te = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        return GraphStats(total_nodes=tn, total_edges=te,
                          node_type_counts=nc, edge_type_counts=ec)


class SecurityKnowledgeGraph:


    def __init__(self):
        self._backend: Neo4jBackend | NetworkXBackend | None = None
        self.backend_type: str = "none"

    def build(self) -> GraphStats:
        with open(DATA_DIR / "owasp_llm_top10.json",  encoding="utf-8") as f:
            owasp_data = json.load(f)
        with open(DATA_DIR / "mitre_attack_llm.json", encoding="utf-8") as f:
            mitre_data = json.load(f)

        driver = _get_driver()
        if driver is not None:
            try:
                backend = Neo4jBackend(driver, _neo4j_cfg["database"])
                stats = backend.build(owasp_data, mitre_data)
                self._backend = backend
                self.backend_type = "neo4j"
                return stats
            except Exception as exc:
                logger.warning("Neo4j build failed (%s) — NetworkX fallback", exc)


        backend = NetworkXBackend()
        stats = backend.build()
        self._backend = backend
        self.backend_type = "networkx"
        return stats

    def match_threats(self, prompt: str, threat_signals: dict[str, float], top_k: int = 8):
        if self._backend is None:
            self.build()
        return self._backend.match_threats(prompt, threat_signals, top_k)

    def get_stats(self) -> GraphStats:
        if self._backend is None:
            self.build()
        return self._backend.get_stats()


    def _get_stats(self) -> GraphStats:
        return self.get_stats()


_kg_instance: SecurityKnowledgeGraph | None = None


def _get_kg() -> SecurityKnowledgeGraph:
    global _kg_instance
    if _kg_instance is None:
        _kg_instance = SecurityKnowledgeGraph()
        _kg_instance.build()
    return _kg_instance


class KnowledgeGraphLayer:


    def __init__(self):
        self._kg: SecurityKnowledgeGraph | None = None

    def _ensure_initialized(self) -> float:
        if self._kg is not None:
            return 0.0
        t0 = time.perf_counter()
        self._kg = _get_kg()
        return (time.perf_counter() - t0) * 1000

    @property
    def backend_type(self) -> str:

        return self._kg.backend_type if self._kg else "none"

    def analyze(
        self,
        prompt:         str,
        threat_signals: dict[str, float],
    ) -> KnowledgeGraphResult:

        t_start = time.perf_counter()
        self._ensure_initialized()

        result = KnowledgeGraphResult(threat_signals=threat_signals)
        result.graph_stats = self._kg.get_stats()


        matches = self._kg.match_threats(prompt, threat_signals)
        result.matched_attacks = matches
        result.total_matches    = len(matches)
        result.top_attack       = matches[0] if matches else None


        result.owasp_hits = list({m.owasp_id for m in matches if m.owasp_id})
        result.mitre_hits = list({m.mitre_id for m in matches if m.mitre_id})


        severity_weights = {"CRITICAL": 1.0, "HIGH": 0.8, "MEDIUM": 0.5, "LOW": 0.2}
        if matches:
            weighted_scores = [
                m.confidence * severity_weights.get(m.severity, 0.5)
                for m in matches[:3]
            ]
            result.kg_risk_score = round(min(sum(weighted_scores) / 3, 1.0), 4)


        if matches:
            top3_names = [m.name for m in matches[:3]]
            result.attack_vector_summary = (
                f"Detected {len(matches)} attack pattern(s). "
                f"Top matches: {', '.join(top3_names[:3])}. "
                f"OWASP categories: {', '.join(result.owasp_hits[:4]) or 'none'}. "
                f"MITRE techniques: {', '.join(result.mitre_hits[:3]) or 'none'}."
            )
        else:
            result.attack_vector_summary = "No attack patterns matched in knowledge graph."

        result.processing_time_ms = round((time.perf_counter() - t_start) * 1000, 2)
        logger.info(
            "KG Layer [%s]: %d matches, owasp=%s, risk=%.3f [%.1fms]",
            self.backend_type,
            result.total_matches, result.owasp_hits,
            result.kg_risk_score, result.processing_time_ms,
        )
        return result
