"""Certificate-variant selection: the minimum deployment set per endpoint.

Given a PEM bundle, each client's trust roots plus the leaf key types and
certificate signature-algorithm OIDs it accepts, each endpoint's candidate
leaf-first fingerprint chains (at most 12), and the client -> endpoint
relationships that must stay connected, this module chooses for every
endpoint the *smallest* set of deployment variants that covers all its
required relationships.

A candidate chain covers a relationship to a client only when **all** of the
following hold:

1. **Hop-by-hop signatures verify.** For every consecutive pair the next
   certificate's public key cryptographically verifies the current one — the
   same signature-verification machinery as ``/scenarios/normalize``
   (issuance is never inferred from subject/issuer name matching).
2. **The terminal cert is trusted.** The chain's final fingerprint is one of
   the client's trust roots.
3. **The leaf key type is accepted.**
4. **Every non-root signature OID is accepted.** The signature algorithm of
   every certificate except the terminating trust root must be in the
   client's accepted OID set (anchors are trusted as-is, so their own
   signature algorithm is never evaluated).

Checks 2–4 are client-specific; check 1 is an intrinsic property of a chain.

The optimization is an *exact* set cover per endpoint (universe = that
endpoint's required clients, sets = each candidate chain's compatible
clients): subsets of chains are tried in increasing cardinality and the
first cover is returned — a per-client greedy heuristic is explicitly not
used. Ties between minimum covers are broken by the lexicographic order of
the full fingerprint chains inside the set (chains sorted, then compared
element by element); each relationship is mapped to the first compatible
variant in that sorted order.

When at least one relationship cannot be covered by any candidate, the whole
request fails with the stable 409 diagnostics produced here: every
uncoverable relationship is listed, sorted by (client_id, endpoint_id), and
for each one every candidate chain carries the (possibly several) reasons it
fails.
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional

from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import dsa, ec, ed25519, ed448, rsa

from .models import (
    NormalizeFingerprintIssue,
    SelectedVariant,
    VariantAssignment,
    VariantChainFailure,
    VariantClient,
    VariantEndpoint,
    VariantEndpointResult,
    VariantSelectionConflict,
    VariantSelectionRequest,
    VariantSelectionResult,
    VariantUncoveredRelationship,
)
from .normalize import (
    FINGERPRINT_RE,
    NormalizeError,
    _build_issuer_edges,
    _load_pem_bundle,
)

# Stable precedence for 409 chain-failure reasons.
REASON_ORDER = {
    "signature_invalid": 0,
    "root_untrusted": 1,
    "leaf_key_type_incompatible": 2,
    "signature_algorithm_incompatible": 3,
}


# ---------------------------------------------------------------------------
# Leaf public-key classification
# ---------------------------------------------------------------------------


def leaf_key_type(cert: x509.Certificate) -> Optional[str]:
    """Classify a certificate's public key as rsa / ec / dsa / ed25519 / ed448.

    Returns None for a key type the selector cannot express (an RSASSA-PSS
    public key is an ``RSAPublicKey`` subclass and is reported as ``rsa``).
    """
    key = cert.public_key()
    if isinstance(key, rsa.RSAPublicKey):
        return "rsa"
    if isinstance(key, ec.EllipticCurvePublicKey):
        return "ec"
    if isinstance(key, dsa.DSAPublicKey):
        return "dsa"
    if isinstance(key, ed25519.Ed25519PublicKey):
        return "ed25519"
    if isinstance(key, getattr(ed448, "Ed448PublicKey", ())):
        return "ed448"
    return None


# ---------------------------------------------------------------------------
# Exact minimum set cover with a lexicographic tie-break
# ---------------------------------------------------------------------------


def minimum_variant_cover(
    chains: list[tuple[str, ...]],
    covers: dict[tuple[str, ...], frozenset[str]],
    required_clients: list[str],
) -> Optional[list[tuple[str, ...]]]:
    """Smallest subset of candidate chains covering every required client.

    ``chains`` are treated as already lexicographically sorted; subsets are
    enumerated in increasing size and lexicographic tuple order, so the first
    feasible subset is the unique minimum-cover tie-break winner: the set's
    chains sorted lexicographically, compared element by element.
    """
    needed = frozenset(required_clients)
    for size in range(1, len(chains) + 1):
        for subset in combinations(chains, size):
            covered: frozenset[str] = frozenset()
            for chain in subset:
                covered |= covers[chain]
            if needed <= covered:
                return list(subset)
    return None


# ---------------------------------------------------------------------------
# Request processing
# ---------------------------------------------------------------------------


def _fingerprint_issue(fp: str, referred_by: str) -> NormalizeFingerprintIssue:
    return NormalizeFingerprintIssue(fingerprint=fp, referred_by=referred_by)


def select_variants(req: VariantSelectionRequest):
    """Compute the minimum per-endpoint variant set, or raise
    :class:`NormalizeError` (HTTP 422) on malformed input.

    Uncoverable relationships are returned as a
    :class:`VariantSelectionConflict` (HTTP 409), not raised.
    """
    # -- fingerprint formatting ------------------------------------------------
    malformed: list[NormalizeFingerprintIssue] = []

    client_roots: dict[str, set[str]] = {}
    for client in req.clients:
        roots: set[str] = set()
        for raw in client.trust_roots:
            fp = raw.strip().lower()
            if not FINGERPRINT_RE.match(fp):
                malformed.append(_fingerprint_issue(
                    raw, f"clients['{client.client_id}'].trust_roots"
                ))
            else:
                roots.add(fp)
        client_roots[client.client_id] = roots

    # Normalized leaf-first fingerprint grid per endpoint. Fingerprints are
    # case-insensitive throughout (pydantic already lower-cased them, but
    # the format is re-validated here to keep the structured NormalizeError
    # 422 contract shared with the other PEM-based routes).
    endpoint_chains: dict[str, list[list[str]]] = {}
    for endpoint in req.endpoints:
        normalized_chains: list[list[str]] = []
        for cidx, chain in enumerate(endpoint.candidate_chains):
            hops: list[str] = []
            for hidx, raw in enumerate(chain):
                fp = raw.strip().lower()
                if not FINGERPRINT_RE.match(fp):
                    malformed.append(_fingerprint_issue(
                        raw,
                        f"endpoints['{endpoint.endpoint_id}']"
                        f".candidate_chains[{cidx}][{hidx}]",
                    ))
                hops.append(fp)
            normalized_chains.append(hops)
        endpoint_chains[endpoint.endpoint_id] = normalized_chains

    if malformed:
        raise NormalizeError(
            "invalid_fingerprint",
            "one or more fingerprints are not 64-character hex SHA-256 values",
            fingerprint_issues=malformed,
        )

    # -- PEM parsing (same dedupe-on-DER-fingerprint rule) ---------------------
    certs, pem_issues = _load_pem_bundle(req.pem_certificates)
    if pem_issues:
        raise NormalizeError(
            "invalid_pem",
            "one or more PEM list elements could not be parsed",
            pem_issues=pem_issues,
        )

    # -- referential integrity: every referenced fingerprint must be in the bag
    missing: list[NormalizeFingerprintIssue] = []
    seen_missing: set[tuple[str, str]] = set()

    def _check_present(fp: str, referred_by: str) -> None:
        if fp not in certs and (fp, referred_by) not in seen_missing:
            seen_missing.add((fp, referred_by))
            missing.append(_fingerprint_issue(fp, referred_by))

    for client in req.clients:
        for fp in sorted(client_roots[client.client_id]):
            _check_present(fp, f"clients['{client.client_id}'].trust_roots")
    for endpoint in req.endpoints:
        for cidx, chain in enumerate(endpoint_chains[endpoint.endpoint_id]):
            for fp in chain:
                _check_present(
                    fp,
                    f"endpoints['{endpoint.endpoint_id}']"
                    f".candidate_chains[{cidx}]",
                )
    if missing:
        raise NormalizeError(
            "unknown_fingerprint",
            "some fingerprints do not match any certificate in the PEM bundle",
            fingerprint_issues=sorted(missing, key=lambda i: (i.referred_by,
                                                              i.fingerprint)),
        )

    # -- issuance graph: one signature verification per ordered pair ----------
    edges = _build_issuer_edges(certs)

    def _signed_by(child_fp: str, issuer_fp: str) -> bool:
        return issuer_fp in edges.get(child_fp, ())

    # -- intrinsic per-chain analysis ------------------------------------------
    chain_info: dict[str, dict[tuple[str, ...], dict]] = {}
    for endpoint in req.endpoints:
        info: dict[tuple[str, ...], dict] = {}
        for chain in endpoint_chains[endpoint.endpoint_id]:
            key = tuple(chain)
            # Every consecutive (child, issuer) pair must verify. A
            # single-cert chain has no hops: its sole certificate is the
            # terminating anchor and is trusted as-is, exactly as the root of
            # a longer chain is never self-signature-checked here.
            signatures_valid = all(
                _signed_by(child, issuer)
                for child, issuer in zip(chain, chain[1:])
            )
            leaf = certs[chain[0]]
            key_type = leaf_key_type(leaf)
            # Signature OIDs of every certificate except the terminal root.
            non_root_oids: list[str] = []
            for fp in chain[:-1]:
                non_root_oids.append(
                    certs[fp].signature_algorithm_oid.dotted_string
                )
            info[key] = {
                "signatures_valid": signatures_valid,
                "leaf_key_type": key_type,
                "non_root_oids": non_root_oids,
            }
        chain_info[endpoint.endpoint_id] = info

    # -- per-relationship coverage ---------------------------------------------
    client_by_id: dict[str, VariantClient] = {c.client_id: c for c in req.clients}
    endpoint_by_id: dict[str, VariantEndpoint] = {
        e.endpoint_id: e for e in req.endpoints
    }
    required_by_endpoint: dict[str, list[str]] = {
        e.endpoint_id: [] for e in req.endpoints
    }
    for rel in req.relationships:
        required_by_endpoint[rel.endpoint_id].append(rel.client_id)

    # covers[endpoint][chain] = required client_ids the chain covers.
    covers: dict[str, dict[tuple[str, ...], set[str]]] = {}
    # failures[(client, endpoint)] lists chains that fail for that client;
    # recorded only when the relationship ends up uncoverable (dropped the
    # moment some chain covers it).
    failures: dict[tuple[str, str], list[VariantChainFailure]] = {}

    for endpoint in req.endpoints:
        ep_covers: dict[tuple[str, ...], set[str]] = {
            tuple(chain): set()
            for chain in endpoint_chains[endpoint.endpoint_id]
        }
        endpoint_id = endpoint.endpoint_id
        for client_id in required_by_endpoint[endpoint_id]:
            client = client_by_id[client_id]
            roots = client_roots[client_id]
            accepted_keys = set(client.acceptable_leaf_key_types)
            accepted_oids = set(client.acceptable_signature_oids)
            chain_failures: list[VariantChainFailure] = []
            covered = False
            for chain in endpoint_chains[endpoint_id]:
                key = tuple(chain)
                info = chain_info[endpoint_id][key]
                reasons: list[str] = []
                if not info["signatures_valid"]:
                    reasons.append("signature_invalid")
                root_fp = chain[-1]
                root_untrusted = root_fp not in roots
                if root_untrusted:
                    reasons.append("root_untrusted")
                key_type = info["leaf_key_type"]
                if key_type not in accepted_keys:
                    reasons.append("leaf_key_type_incompatible")
                bad_oids = sorted({
                    oid for oid in info["non_root_oids"]
                    if oid not in accepted_oids
                })
                if bad_oids:
                    reasons.append("signature_algorithm_incompatible")
                if not reasons:
                    covered = True
                    ep_covers[key].add(client_id)
                else:
                    chain_failures.append(VariantChainFailure(
                        chain=list(chain),
                        reasons=sorted(reasons, key=REASON_ORDER.get),
                        leaf_key_type=key_type,
                        untrusted_root_fingerprint=root_fp
                        if root_untrusted else None,
                        unacceptable_signature_oids=bad_oids,
                    ))
            if not covered:
                failures[(client_id, endpoint_id)] = chain_failures
        covers[endpoint_id] = ep_covers

    # -- 409: every uncoverable relationship, stably ordered -------------------
    uncoverable: list[VariantUncoveredRelationship] = []
    for client_id, endpoint_id in sorted(
        (c, e) for e, cs in required_by_endpoint.items() for c in cs
    ):
        chain_failures = failures.get((client_id, endpoint_id))
        if chain_failures is None:
            continue
        chain_failures = sorted(
            chain_failures, key=lambda f: tuple(f.chain)
        )
        uncoverable.append(VariantUncoveredRelationship(
            client_id=client_id,
            endpoint_id=endpoint_id,
            candidate_failures=chain_failures,
        ))
    if uncoverable:
        first = uncoverable[0]
        return VariantSelectionConflict(
            message=(
                f"{len(uncoverable)} client->endpoint relationship(s) cannot "
                f"be covered by any candidate variant; first: "
                f"{first.client_id} -> {first.endpoint_id} "
                f"({len(first.candidate_failures)} candidate chain(s) all "
                f"rejected)"
            ),
            uncoverable_relationships=uncoverable,
        )

    # -- exact minimum set cover per endpoint + assignments --------------------
    results: list[VariantEndpointResult] = []
    for endpoint in sorted(endpoint_by_id, key=lambda eid: eid):
        sorted_chains = sorted(
            {tuple(c) for c in endpoint_chains[endpoint]}
        )
        required_clients = sorted(set(required_by_endpoint[endpoint]))
        # An endpoint with no required relationships needs no deployment
        # variant at all.
        if not required_clients:
            results.append(VariantEndpointResult(
                endpoint_id=endpoint,
                variant_count=0,
                selected_chains=[],
                variants=[],
                assignments=[],
            ))
            continue
        selected = minimum_variant_cover(
            sorted_chains, covers[endpoint], required_clients
        )
        # Uncoverable relationships were reported above; every endpoint that
        # reaches this point has at least one chain per required client.
        assert selected is not None

        variants = [
            SelectedVariant(
                chain=list(chain),
                leaf_key_type=chain_info[endpoint][chain]["leaf_key_type"],
            )
            for chain in selected
        ]
        assignments = [
            VariantAssignment(
                client_id=client_id,
                chain=list(next(
                    variant for variant in selected
                    if client_id in covers[endpoint][variant]
                )),
            )
            for client_id in required_clients
        ]
        results.append(VariantEndpointResult(
            endpoint_id=endpoint,
            variant_count=len(selected),
            selected_chains=[list(variant) for variant in selected],
            variants=variants,
            assignments=assignments,
        ))

    return VariantSelectionResult(results=results)
