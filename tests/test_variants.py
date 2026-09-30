"""Tests for POST /deployments/variants (minimum certificate-variant set)."""
from __future__ import annotations

import hashlib
import itertools
from datetime import datetime, timezone

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa
from cryptography.x509.oid import NameOID
from fastapi.testclient import TestClient

from app.main import app
from app.variants import leaf_key_type, minimum_variant_cover

NOW = datetime(2026, 9, 1, tzinfo=timezone.utc)
NOT_BEFORE = datetime(2024, 1, 1, tzinfo=timezone.utc)
NOT_AFTER = datetime(2035, 1, 1, tzinfo=timezone.utc)

# Signature OIDs.
SHA256_RSA = "1.2.840.113549.1.1.11"
SHA384_RSA = "1.2.840.113549.1.1.12"
ECDSA_SHA256 = "1.2.840.10045.4.3.2"
ED25519_OID = "1.3.101.112"

client = TestClient(app)


# ---------------------------------------------------------------------------
# PKI factory
# ---------------------------------------------------------------------------


def _dn(cn):
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def _issue(subject_cn, issuer_cn, issuer_key, subject_key, *,
           hash_alg=hashes.SHA256(), is_ca=False):
    builder = (
        x509.CertificateBuilder()
        .subject_name(_dn(subject_cn))
        .issuer_name(_dn(issuer_cn))
        .public_key(subject_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(NOT_BEFORE)
        .not_valid_after(NOT_AFTER)
        .add_extension(
            x509.BasicConstraints(ca=is_ca, path_length=None), critical=True
        )
    )
    # PureEdDSA keys sign without a hash parameter (the key fixes the OID).
    if isinstance(issuer_key, ed25519.Ed25519PrivateKey):
        return builder.sign(issuer_key, None)
    return builder.sign(issuer_key, hash_alg)


def _root(cn, key):
    return _issue(cn, cn, key, key, is_ca=True)


def pem(cert) -> str:
    return cert.public_bytes(serialization.Encoding.PEM).decode()


def fp(cert) -> str:
    return hashlib.sha256(
        cert.public_bytes(serialization.Encoding.DER)
    ).hexdigest()


@pytest.fixture
def pki():
    """Two independent PKI hierarchies plus a third (kiosk) root.

    legacy root (RSA key)   : legacy-client trust
        legacy-intermediate (RSA key, signs with SHA-256)
            leaf-rsa-sha384(RSA leaf signed with SHA-384: an OID the
                            modern client does not accept)
            leaf-rsa-sha256(RSA leaf signed with SHA-256)
    modern root (EC key)    : modern-client trust
        modern-intermediate (EC key, ECDSA-SHA256)
            leaf-ec         (EC leaf)
    ed root (Ed25519 key)   : also trusted by the modern client
        leaf-ed             (Ed25519 leaf signed with Ed25519)
    kiosk root (EC key)     : only the picky client trusts it
        leaf-kiosk          (EC leaf)
    """
    legacy_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    legacy_root = _root("Legacy Root CA", legacy_key)

    inter_rsa_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048
    )
    inter_rsa = _issue(
        "Legacy Intermediate CA", "Legacy Root CA", legacy_key,
        inter_rsa_key, is_ca=True,
    )
    leaf_rsa_sha384 = _issue(
        "api.example.com", "Legacy Intermediate CA", inter_rsa_key,
        rsa.generate_private_key(65537, 2048), hash_alg=hashes.SHA384(),
    )
    leaf_rsa_sha256 = _issue(
        "api.example.com", "Legacy Intermediate CA", inter_rsa_key,
        rsa.generate_private_key(65537, 2048),
    )

    modern_key = ec.generate_private_key(ec.SECP256R1())
    modern_root = _root("Modern Root CA", modern_key)

    inter_ec_key = ec.generate_private_key(ec.SECP256R1())
    inter_ec = _issue(
        "Modern Intermediate CA", "Modern Root CA", modern_key,
        inter_ec_key, is_ca=True,
    )
    leaf_ec = _issue(
        "api.example.com", "Modern Intermediate CA", inter_ec_key,
        ec.generate_private_key(ec.SECP256R1()),
    )
    ed_key = ed25519.Ed25519PrivateKey.generate()
    ed_root = _root("Ed Root CA", ed_key)
    leaf_ed = _issue(
        "api.example.com", "Ed Root CA", ed_key,
        ed25519.Ed25519PrivateKey.generate(),
    )

    kiosk_key = ec.generate_private_key(ec.SECP256R1())
    kiosk_root = _root("Kiosk Root CA", kiosk_key)
    leaf_kiosk = _issue(
        "api.example.com", "Kiosk Root CA", kiosk_key,
        ec.generate_private_key(ec.SECP256R1()),
    )

    # A third independent EC root, used by the greedy-trap test.
    grid_key = ec.generate_private_key(ec.SECP256R1())
    grid_root = _root("Grid Root CA", grid_key)
    leaf_grid = _issue(
        "api.example.com", "Grid Root CA", grid_key,
        ec.generate_private_key(ec.SECP256R1()),
    )

    return {
        "legacy_root": legacy_root,
        "inter_rsa": inter_rsa,
        "leaf_rsa_sha384": leaf_rsa_sha384,
        "leaf_rsa_sha256": leaf_rsa_sha256,
        "modern_root": modern_root,
        "inter_ec": inter_ec,
        "leaf_ec": leaf_ec,
        "ed_root": ed_root,
        "leaf_ed": leaf_ed,
        "kiosk_root": kiosk_root,
        "leaf_kiosk": leaf_kiosk,
        "grid_root": grid_root,
        "leaf_grid": leaf_grid,
    }


def chain(*certs) -> list[str]:
    return [fp(c) for c in certs]


def post(payload):
    return client.post("/deployments/variants", json=payload)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestLeafKeyType:
    def test_classification(self, pki):
        assert leaf_key_type(pki["leaf_rsa_sha256"]) == "rsa"
        assert leaf_key_type(pki["leaf_ec"]) == "ec"
        assert leaf_key_type(pki["leaf_ed"]) == "ed25519"
        assert leaf_key_type(pki["legacy_root"]) == "rsa"
        assert leaf_key_type(pki["modern_root"]) == "ec"


class TestMinimumVariantCover:
    def test_exact_minimum_beats_greedy(self):
        # Classic greedy counter-example:
        # S1 covers {c2,c3}, S2 covers {c1,c2}, S3 covers {c1}; greedy picks
        # S1 first and needs 3 sets, while {S2, S1} covers with 2.
        c = {
            ("S1",): frozenset({"c2", "c3"}),
            ("S2",): frozenset({"c1", "c2"}),
            ("S3",): frozenset({"c1"}),
        }
        chains = sorted(c)
        result = minimum_variant_cover(chains, c, ["c1", "c2", "c3"])
        assert result == [("S1",), ("S2",)]

    def test_uncoverable_returns_none(self):
        c = {("S1",): frozenset({"c1"})}
        assert minimum_variant_cover(sorted(c), c, ["c1", "c2"]) is None

    def test_lexicographic_tie_break(self):
        # Both {A, B} and {A, C} are minimum covers; sorted tuple order
        # decides.
        c = {
            ("A",): frozenset({"x"}),
            ("B",): frozenset({"y"}),
            ("C",): frozenset({"y"}),
        }
        result = minimum_variant_cover(sorted(c), c, ["x", "y"])
        assert result == [("A",), ("B",)]

    def test_full_chain_lexicographic_order_is_elementwise(self):
        # Two chains share a leaf fingerprint but differ at the root hop;
        # the comparison walks element by element over the full chains.
        c = {
            ("leaf", "root-a"): frozenset({"x"}),
            ("leaf", "root-b"): frozenset({"y"}),
            ("only-y",): frozenset({"y"}),
        }
        chains = sorted(c)
        assert chains == [("leaf", "root-a"), ("leaf", "root-b"),
                          ("only-y",)]
        result = minimum_variant_cover(
            chains, c, ["x", "y"]
        )
        assert result == [("leaf", "root-a"), ("leaf", "root-b")]


# ---------------------------------------------------------------------------
# End-to-end happy path
# ---------------------------------------------------------------------------


def _modern_client(pki):
    return {
        "client_id": "modern",
        "trust_roots": [fp(pki["modern_root"]), fp(pki["ed_root"])],
        "acceptable_leaf_key_types": ["rsa", "ec", "ed25519"],
        "acceptable_signature_oids": [SHA256_RSA, ECDSA_SHA256, ED25519_OID],
    }


def _legacy_client(pki):
    # Legacy accepts only sha256WithRSAEncryption: the SHA-384 leaf is a
    # valid chain cryptographically but rejected on the signature OID.
    return {
        "client_id": "legacy",
        "trust_roots": [fp(pki["legacy_root"])],
        "acceptable_leaf_key_types": ["rsa"],
        "acceptable_signature_oids": [SHA256_RSA],
    }


def _bundle(pki):
    return [pem(c) for c in pki.values()]


class TestSelectionHappyPath:
    def test_two_clients_share_rsa_variant(self, pki):
        # leaf-rsa-sha256 -> inter-rsa -> legacy root covers legacy;
        # leaf-ec -> inter-ec -> modern root covers modern;
        # the RSA chain is also accepted by modern (rsa key, SHA256 RSA
        # signatures, but the root is not modern's) so no sharing: two
        # variants are still required because roots differ.
        payload = {
            "pem_certificates": _bundle(pki),
            "clients": [_modern_client(pki), _legacy_client(pki)],
            "endpoints": [{
                "endpoint_id": "api.example.com",
                "candidate_chains": [
                    chain(pki["leaf_rsa_sha384"], pki["inter_rsa"],
                          pki["legacy_root"]),
                    chain(pki["leaf_rsa_sha256"], pki["inter_rsa"],
                          pki["legacy_root"]),
                    chain(pki["leaf_ec"], pki["inter_ec"],
                          pki["modern_root"]),
                ],
            }],
            "relationships": [
                {"client_id": "legacy", "endpoint_id": "api.example.com"},
                {"client_id": "modern", "endpoint_id": "api.example.com"},
            ],
        }
        r = post(payload)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "ok"
        result = body["results"][0]
        assert result["endpoint_id"] == "api.example.com"
        assert result["variant_count"] == 2
        assert result["selected_chains"] == sorted(result["selected_chains"])
        expected = {
            tuple(chain(pki["leaf_rsa_sha256"], pki["inter_rsa"],
                        pki["legacy_root"])),
            tuple(chain(pki["leaf_ec"], pki["inter_ec"],
                        pki["modern_root"])),
        }
        assert {tuple(c) for c in result["selected_chains"]} == expected
        # leaf_rsa_sha384 is a cryptographically valid chain but legacy
        # does not accept its signature OID; it is simply never selected.
        assignments = {a["client_id"]: tuple(a["chain"])
                       for a in result["assignments"]}
        assert assignments["legacy"] == tuple(
            chain(pki["leaf_rsa_sha256"], pki["inter_rsa"],
                  pki["legacy_root"])
        )
        assert assignments["modern"] == tuple(
            chain(pki["leaf_ec"], pki["inter_ec"], pki["modern_root"])
        )
        assert [a["client_id"] for a in result["assignments"]] == [
            "legacy", "modern"
        ]
        key_types = {tuple(v["chain"]): v["leaf_key_type"]
                     for v in result["variants"]}
        assert key_types[
            tuple(chain(pki["leaf_rsa_sha256"], pki["inter_rsa"],
                        pki["legacy_root"]))
        ] == "rsa"
        assert key_types[
            tuple(chain(pki["leaf_ec"], pki["inter_ec"],
                        pki["modern_root"]))
        ] == "ec"

    def test_shared_chain_covers_both_clients_with_one_variant(self, pki):
        # Make modern also trust the legacy root: the RSA-SHA256 chain then
        # covers both clients, so the optimum collapses to one variant.
        modern = _modern_client(pki)
        modern["trust_roots"].append(fp(pki["legacy_root"]))
        payload = {
            "pem_certificates": _bundle(pki),
            "clients": [modern, _legacy_client(pki)],
            "endpoints": [{
                "endpoint_id": "api.example.com",
                "candidate_chains": [
                    chain(pki["leaf_ec"], pki["inter_ec"],
                          pki["modern_root"]),
                    chain(pki["leaf_rsa_sha256"], pki["inter_rsa"],
                          pki["legacy_root"]),
                ],
            }],
            "relationships": [
                {"client_id": "modern", "endpoint_id": "api.example.com"},
                {"client_id": "legacy", "endpoint_id": "api.example.com"},
            ],
        }
        r = post(payload)
        assert r.status_code == 200, r.text
        result = r.json()["results"][0]
        assert result["variant_count"] == 1
        assert result["selected_chains"] == [
            chain(pki["leaf_rsa_sha256"], pki["inter_rsa"],
                  pki["legacy_root"])
        ]
        for assignment in result["assignments"]:
            assert assignment["chain"] == chain(
                pki["leaf_rsa_sha256"], pki["inter_rsa"], pki["legacy_root"]
            )

    def test_results_sorted_by_endpoint_and_zero_for_unrelated(self, pki):
        payload = {
            "pem_certificates": _bundle(pki),
            "clients": [_modern_client(pki)],
            "endpoints": [
                {"endpoint_id": "zzz.example.com", "candidate_chains": [
                    chain(pki["leaf_ec"], pki["inter_ec"],
                          pki["modern_root"]),
                ]},
                {"endpoint_id": "aaa.example.com", "candidate_chains": [
                    chain(pki["leaf_ed"], pki["ed_root"]),
                ]},
            ],
            "relationships": [
                {"client_id": "modern", "endpoint_id": "zzz.example.com"},
            ],
        }
        r = post(payload)
        assert r.status_code == 200, r.text
        results = r.json()["results"]
        assert [x["endpoint_id"] for x in results] == [
            "aaa.example.com", "zzz.example.com"
        ]
        aaa = results[0]
        assert aaa["variant_count"] == 0
        assert aaa["selected_chains"] == []
        assert aaa["assignments"] == []

    def test_endpoints_solved_independently(self, pki):
        # The same client relates to two endpoints; each endpoint's cover is
        # computed on its own candidate set.
        modern = _modern_client(pki)
        legacy = _legacy_client(pki)
        payload = {
            "pem_certificates": _bundle(pki),
            "clients": [modern, legacy],
            "endpoints": [
                {"endpoint_id": "a.example.com", "candidate_chains": [
                    chain(pki["leaf_rsa_sha256"], pki["inter_rsa"],
                          pki["legacy_root"]),
                    chain(pki["leaf_ec"], pki["inter_ec"],
                          pki["modern_root"]),
                ]},
                {"endpoint_id": "b.example.com", "candidate_chains": [
                    chain(pki["leaf_ed"], pki["ed_root"]),
                ]},
            ],
            "relationships": [
                {"client_id": "legacy", "endpoint_id": "a.example.com"},
                {"client_id": "modern", "endpoint_id": "a.example.com"},
                {"client_id": "modern", "endpoint_id": "b.example.com"},
            ],
        }
        r = post(payload)
        assert r.status_code == 200, r.text
        by_ep = {x["endpoint_id"]: x for x in r.json()["results"]}
        assert by_ep["a.example.com"]["variant_count"] == 2
        assert by_ep["b.example.com"]["variant_count"] == 1
        assert by_ep["b.example.com"]["selected_chains"] == [
            chain(pki["leaf_ed"], pki["ed_root"])
        ]

    def test_endpoint_with_rsa_only_candidate_still_feasible_for_modern(self, pki):
        # modern accepts RSA + the SHA-256 RSA OID but does not trust the
        # legacy root; adding the root to its trust set makes the single RSA
        # variant cover every client of that endpoint (exact optimum 1).
        modern = _modern_client(pki)
        modern["trust_roots"].append(fp(pki["legacy_root"]))
        payload = {
            "pem_certificates": _bundle(pki),
            "clients": [modern, _legacy_client(pki)],
            "endpoints": [{"endpoint_id": "only-rsa", "candidate_chains": [
                chain(pki["leaf_rsa_sha256"], pki["inter_rsa"],
                      pki["legacy_root"]),
            ]}],
            "relationships": [
                {"client_id": "legacy", "endpoint_id": "only-rsa"},
                {"client_id": "modern", "endpoint_id": "only-rsa"},
            ],
        }
        r = post(payload)
        assert r.status_code == 200, r.text
        result = r.json()["results"][0]
        assert result["variant_count"] == 1
        assert len(result["assignments"]) == 2

    def test_assignment_picks_first_compatible_in_sorted_order(self, pki):
        # Two distinct EC chains both cover modern (modern root is trusted by
        # both); assignment must map modern to whichever full chain sorts
        # first, independently of the input order.
        chains = [
            chain(pki["leaf_ed"], pki["ed_root"]),
            chain(pki["leaf_ec"], pki["inter_ec"], pki["modern_root"]),
        ]
        # leaf_ed chain cannot cover modern unless ed root is trusted: the
        # fixture modern trusts both modern and ed roots.
        payload = {
            "pem_certificates": _bundle(pki),
            "clients": [_modern_client(pki)],
            "endpoints": [{
                "endpoint_id": "tie.example.com",
                "candidate_chains": list(reversed(chains)),
            }],
            "relationships": [
                {"client_id": "modern", "endpoint_id": "tie.example.com"},
            ],
        }
        r = post(payload)
        assert r.status_code == 200, r.text
        result = r.json()["results"][0]
        assert result["variant_count"] == 1
        first = sorted(chains)[0]
        assert result["selected_chains"][0] == first
        assert result["assignments"][0]["chain"] == first


        # A chain consisting solely of a trusted fingerprint has no hops to
        # verify; it is accepted on trust alone.
        payload = {
            "pem_certificates": _bundle(pki),
            "clients": [{
                "client_id": "modern",
                "trust_roots": [fp(pki["modern_root"])],
                "acceptable_leaf_key_types": ["ec"],
                "acceptable_signature_oids": [ECDSA_SHA256],
            }],
            "endpoints": [{
                "endpoint_id": "root-serve",
                "candidate_chains": [[fp(pki["modern_root"])]],
            }],
            "relationships": [
                {"client_id": "modern", "endpoint_id": "root-serve"},
            ],
        }
        r = post(payload)
        assert r.status_code == 200, r.text
        result = r.json()["results"][0]
        assert result["variant_count"] == 1
        assert result["selected_chains"] == [[fp(pki["modern_root"])]]
        assert result["variants"][0]["leaf_key_type"] == "ec"


# ---------------------------------------------------------------------------
# 409: uncoverable relationships with full diagnostics
# ---------------------------------------------------------------------------


class TestUncoverable409:
    def test_all_reasons_reported_independently(self, pki):
        # picky trusts nobody relevant (only the kiosk root, which never
        # appears below), accepts EC leaves and only the ECDSA-SHA256 OID.
        picky = {
            "client_id": "picky",
            "trust_roots": [fp(pki["kiosk_root"])],
            "acceptable_leaf_key_types": ["ec"],
            "acceptable_signature_oids": [ECDSA_SHA256],
        }
        # A signature-forged chain: leaf-ec spliced under the RSA
        # intermediate -> hop verification fails for every client.
        forged = chain(pki["leaf_ec"], pki["inter_rsa"], pki["legacy_root"])
        payload = {
            "pem_certificates": _bundle(pki),
            "clients": [picky],
            "endpoints": [{
                "endpoint_id": "api.example.com",
                "candidate_chains": [
                    chain(pki["leaf_rsa_sha384"], pki["inter_rsa"],
                          pki["legacy_root"]),
                    chain(pki["leaf_ec"], pki["inter_ec"],
                          pki["modern_root"]),
                    forged,
                ],
            }],
            "relationships": [
                {"client_id": "picky", "endpoint_id": "api.example.com"},
            ],
        }
        r = post(payload)
        assert r.status_code == 409, r.text
        body = r.json()
        assert body["status"] == "infeasible"
        assert body["reason"] == "relationship_uncoverable"
        rel = body["uncoverable_relationships"][0]
        assert rel["client_id"] == "picky"
        assert rel["endpoint_id"] == "api.example.com"
        by_leaf = {tuple(f["chain"]): f for f in rel["candidate_failures"]}
        # chains sorted lexicographically
        chains = [tuple(f["chain"]) for f in rel["candidate_failures"]]
        assert chains == sorted(chains)

        sha384_fail = by_leaf[
            tuple(chain(pki["leaf_rsa_sha384"], pki["inter_rsa"],
                        pki["legacy_root"]))
        ]
        assert sha384_fail["reasons"] == [
            "root_untrusted",
            "leaf_key_type_incompatible",
            "signature_algorithm_incompatible",
        ]
        assert sha384_fail["leaf_key_type"] == "rsa"
        assert sha384_fail["untrusted_root_fingerprint"] == fp(
            pki["legacy_root"]
        )
        assert sha384_fail["unacceptable_signature_oids"] == sorted(
            {SHA384_RSA, SHA256_RSA}
        )

        ec_fail = by_leaf[
            tuple(chain(pki["leaf_ec"], pki["inter_ec"], pki["modern_root"]))
        ]
        assert ec_fail["reasons"] == ["root_untrusted"]
        assert ec_fail["leaf_key_type"] == "ec"
        assert ec_fail["untrusted_root_fingerprint"] == fp(
            pki["modern_root"]
        )
        assert ec_fail["unacceptable_signature_oids"] == []

        forged_fail = by_leaf[tuple(forged)]
        # Both non-root certs of the forged chain are RSA-signed (OID
        # sha256WithRSAEncryption), which picky does not accept either.
        assert forged_fail["reasons"] == [
            "signature_invalid",
            "root_untrusted",
            "signature_algorithm_incompatible",
        ]
        assert forged_fail["leaf_key_type"] == "ec"
        assert forged_fail["untrusted_root_fingerprint"] == fp(
            pki["legacy_root"]
        )
        assert forged_fail["unacceptable_signature_oids"] == [SHA256_RSA]

        assert "picky" in body["message"]

    def test_kiosk_chain_covers_picky(self, pki):
        # Complement of the guard above: trusting the kiosk root makes the
        # kiosk chain a valid cover -> 200 with one variant.
        picky = {
            "client_id": "picky",
            "trust_roots": [fp(pki["kiosk_root"])],
            "acceptable_leaf_key_types": ["ec"],
            "acceptable_signature_oids": [ECDSA_SHA256],
        }
        payload = {
            "pem_certificates": _bundle(pki),
            "clients": [picky],
            "endpoints": [{
                "endpoint_id": "api.example.com",
                "candidate_chains": [
                    chain(pki["leaf_ec"], pki["inter_ec"],
                          pki["modern_root"]),
                    chain(pki["leaf_kiosk"], pki["kiosk_root"]),
                ],
            }],
            "relationships": [
                {"client_id": "picky", "endpoint_id": "api.example.com"},
            ],
        }
        r = post(payload)
        assert r.status_code == 200, r.text
        assert r.json()["results"][0]["selected_chains"] == [
            chain(pki["leaf_kiosk"], pki["kiosk_root"])
        ]

    def test_signature_invalid_chain_covers_nobody(self, pki):
        # Splice leaf-ec under the RSA intermediate: the leaf->intermediate
        # hop signature does not verify.
        modern = _modern_client(pki)
        modern["trust_roots"].append(fp(pki["legacy_root"]))
        payload = {
            "pem_certificates": _bundle(pki),
            "clients": [modern],
            "endpoints": [{
                "endpoint_id": "api.example.com",
                "candidate_chains": [
                    chain(pki["leaf_ec"], pki["inter_rsa"],
                          pki["legacy_root"]),
                ],
            }],
            "relationships": [
                {"client_id": "modern", "endpoint_id": "api.example.com"},
            ],
        }
        r = post(payload)
        assert r.status_code == 409
        failure = r.json()["uncoverable_relationships"][0][
            "candidate_failures"
        ][0]
        assert "signature_invalid" in failure["reasons"]
        assert "root_untrusted" not in failure["reasons"]

    def test_exact_optimum_beats_per_client_greedy_end_to_end(self, pki):
        # A genuine greedy-trap set system (verified by exhaustive search):
        # four clients, three EC/ECDSA chains terminating at three different
        # roots; every client accepts every chain cryptographically and
        # differs only in which roots it trusts:
        #   chain-A (leaf_kiosk -> kiosk root)        covers {c1, c2}
        #   chain-B (leaf_ec -> inter -> modern root) covers {c1, c3}
        #   chain-C (leaf_grid -> grid root)          covers {c2, c4}
        # The unique 2-variant cover is {B, C} (union = all four clients).
        # A largest-set-first per-client greedy that breaks the A/B size tie
        # towards A commits to A, then still needs B for c3 and C for c4 ->
        # 3 variants; the exact solver must return 2.
        ec_only = ["ec"]
        ec_oids = [ECDSA_SHA256]
        clients = [
            {"client_id": "c1",
             "trust_roots": [fp(pki["kiosk_root"]), fp(pki["modern_root"])],
             "acceptable_leaf_key_types": ec_only,
             "acceptable_signature_oids": ec_oids},
            {"client_id": "c2",
             "trust_roots": [fp(pki["kiosk_root"]), fp(pki["grid_root"])],
             "acceptable_leaf_key_types": ec_only,
             "acceptable_signature_oids": ec_oids},
            {"client_id": "c3",
             "trust_roots": [fp(pki["modern_root"])],
             "acceptable_leaf_key_types": ec_only,
             "acceptable_signature_oids": ec_oids},
            {"client_id": "c4",
             "trust_roots": [fp(pki["grid_root"])],
             "acceptable_leaf_key_types": ec_only,
             "acceptable_signature_oids": ec_oids},
        ]
        chain_a = chain(pki["leaf_kiosk"], pki["kiosk_root"])
        chain_b = chain(pki["leaf_ec"], pki["inter_ec"], pki["modern_root"])
        chain_c = chain(pki["leaf_grid"], pki["grid_root"])
        chains = [chain_a, chain_b, chain_c]
        payload = {
            "pem_certificates": _bundle(pki),
            "clients": clients,
            "endpoints": [{
                "endpoint_id": "greedy-trap",
                "candidate_chains": chains,
            }],
            "relationships": [
                {"client_id": f"c{i}", "endpoint_id": "greedy-trap"}
                for i in (1, 2, 3, 4)
            ],
        }
        r = post(payload)
        assert r.status_code == 200, r.text
        result = r.json()["results"][0]
        assert result["variant_count"] == 2
        assert result["selected_chains"] == sorted([chain_b, chain_c])
        assignment = {a["client_id"]: tuple(a["chain"])
                      for a in result["assignments"]}
        assert assignment["c1"] == tuple(chain_b)
        assert assignment["c2"] == tuple(chain_c)
        assert assignment["c3"] == tuple(chain_b)
        assert assignment["c4"] == tuple(chain_c)
        # The optimum must not depend on input order: reversed chains give
        # the same selected set.
        payload["endpoints"][0]["candidate_chains"] = list(reversed(chains))
        r2 = post(payload)
        assert r2.status_code == 200
        assert r2.json()["results"][0]["selected_chains"] == \
            result["selected_chains"]

    def test_signature_invalid_at_intermediate_root_hop(self, pki):
        # The leaf is properly signed by the EC intermediate, but the
        # intermediate is spliced onto the legacy root: that second hop does
        # not verify even though the first one does.
        modern = _modern_client(pki)
        modern["trust_roots"].append(fp(pki["legacy_root"]))
        forged = chain(pki["leaf_ec"], pki["inter_ec"],
                       pki["legacy_root"])
        payload = {
            "pem_certificates": _bundle(pki),
            "clients": [modern],
            "endpoints": [{
                "endpoint_id": "api.example.com",
                "candidate_chains": [forged],
            }],
            "relationships": [
                {"client_id": "modern", "endpoint_id": "api.example.com"},
            ],
        }
        r = post(payload)
        assert r.status_code == 409
        failure = r.json()["uncoverable_relationships"][0][
            "candidate_failures"
        ][0]
        assert failure["reasons"] == ["signature_invalid"]
        assert failure["leaf_key_type"] == "ec"


    def test_legacy_rejects_ec_chain_with_multiple_reasons(self, pki):
        # With the SHA-256 RSA chain present, legacy is covered -> 200.
        ok_payload = {
            "pem_certificates": _bundle(pki),
            "clients": [_legacy_client(pki)],
            "endpoints": [{
                "endpoint_id": "api.example.com",
                "candidate_chains": [
                    chain(pki["leaf_rsa_sha256"], pki["inter_rsa"],
                          pki["legacy_root"]),
                ],
            }],
            "relationships": [
                {"client_id": "legacy", "endpoint_id": "api.example.com"},
            ],
        }
        assert post(ok_payload).status_code == 200

        # Without any compatible RSA chain: every candidate fails for
        # independent reasons, all reported together.
        payload2 = {
            "pem_certificates": _bundle(pki),
            "clients": [_legacy_client(pki)],
            "endpoints": [{
                "endpoint_id": "api.example.com",
                "candidate_chains": [
                    chain(pki["leaf_ec"], pki["inter_ec"],
                          pki["modern_root"]),
                    chain(pki["leaf_ed"], pki["ed_root"]),
                ],
            }],
            "relationships": [
                {"client_id": "legacy", "endpoint_id": "api.example.com"},
            ],
        }
        r2 = post(payload2)
        assert r2.status_code == 409
        rel = r2.json()["uncoverable_relationships"][0]
        by_leaf = {tuple(f["chain"]): f for f in rel["candidate_failures"]}
        ec_fail = by_leaf[
            tuple(chain(pki["leaf_ec"], pki["inter_ec"], pki["modern_root"]))
        ]
        assert ec_fail["reasons"] == [
            "root_untrusted",
            "leaf_key_type_incompatible",
            "signature_algorithm_incompatible",
        ]
        assert ec_fail["unacceptable_signature_oids"] == [ECDSA_SHA256]
        ed_fail = by_leaf[tuple(chain(pki["leaf_ed"], pki["ed_root"]))]
        # leaf_ed directly under modern root: only one non-root cert (leaf),
        # signed with Ed25519.
        assert ed_fail["reasons"] == [
            "root_untrusted",
            "leaf_key_type_incompatible",
            "signature_algorithm_incompatible",
        ]
        assert ed_fail["unacceptable_signature_oids"] == [ED25519_OID]

    def test_uncoverable_relationships_stably_sorted(self, pki):
        modern = _modern_client(pki)
        legacy = _legacy_client(pki)
        # Add a second endpoint whose only candidate is bad for both clients.
        payload = {
            "pem_certificates": _bundle(pki),
            "clients": [modern, legacy],
            "endpoints": [
                {
                    "endpoint_id": "api.example.com",
                    "candidate_chains": [
                        chain(pki["leaf_ec"], pki["inter_ec"],
                              pki["modern_root"]),
                    ],
                },
                {
                    "endpoint_id": "cdn.example.com",
                    "candidate_chains": [
                        chain(pki["leaf_ed"], pki["ed_root"]),
                    ],
                },
            ],
            "relationships": [
                {"client_id": "modern", "endpoint_id": "api.example.com"},
                {"client_id": "legacy", "endpoint_id": "api.example.com"},
                {"client_id": "legacy", "endpoint_id": "cdn.example.com"},
            ],
        }
        r = post(payload)
        assert r.status_code == 409
        pairs = [
            (x["client_id"], x["endpoint_id"])
            for x in r.json()["uncoverable_relationships"]
        ]
        assert pairs == sorted(pairs)
        assert pairs == [
            ("legacy", "api.example.com"),
            ("legacy", "cdn.example.com"),
        ]
        # The modern -> api relationship is coverable and must not appear.
        assert ("modern", "api.example.com") not in pairs


# ---------------------------------------------------------------------------
# 422: malformed requests
# ---------------------------------------------------------------------------


class TestValidation422:
    def _valid_payload(self, pki):
        return {
            "pem_certificates": _bundle(pki),
            "clients": [_modern_client(pki)],
            "endpoints": [{
                "endpoint_id": "api.example.com",
                "candidate_chains": [
                    chain(pki["leaf_ec"], pki["inter_ec"],
                          pki["modern_root"]),
                ],
            }],
            "relationships": [
                {"client_id": "modern", "endpoint_id": "api.example.com"},
            ],
        }

    def test_invalid_pem(self, pki):
        payload = self._valid_payload(pki)
        payload["pem_certificates"] = ["not a pem at all"]
        r = post(payload)
        assert r.status_code == 422
        assert r.json()["detail"]["error"] == "invalid_pem"

    def test_pem_dedup_still_applies(self, pki):
        payload = self._valid_payload(pki)
        payload["pem_certificates"] = payload["pem_certificates"] * 2
        r = post(payload)
        assert r.status_code == 200, r.text

    def test_invalid_fingerprint_in_chain(self, pki):
        payload = self._valid_payload(pki)
        payload["endpoints"][0]["candidate_chains"][0][0] = "zz"
        r = post(payload)
        assert r.status_code == 422
        assert r.json()["detail"]["error"] == "invalid_fingerprint"

    def test_unknown_fingerprint(self, pki):
        payload = self._valid_payload(pki)
        phantom = "a" * 64
        payload["endpoints"][0]["candidate_chains"][0][0] = phantom
        r = post(payload)
        assert r.status_code == 422
        assert r.json()["detail"]["error"] == "unknown_fingerprint"

    def test_unknown_trust_root_fingerprint(self, pki):
        payload = self._valid_payload(pki)
        payload["clients"][0]["trust_roots"] = ["b" * 64]
        r = post(payload)
        assert r.status_code == 422
        assert r.json()["detail"]["error"] == "unknown_fingerprint"

    def test_too_many_candidate_chains(self, pki):
        payload = self._valid_payload(pki)
        good = payload["endpoints"][0]["candidate_chains"][0]
        payload["endpoints"][0]["candidate_chains"] = [good] * 13
        r = post(payload)
        assert r.status_code == 422

    def test_duplicate_candidate_chains_rejected(self, pki):
        payload = self._valid_payload(pki)
        good = payload["endpoints"][0]["candidate_chains"][0]
        payload["endpoints"][0]["candidate_chains"] = [good, list(good)]
        r = post(payload)
        assert r.status_code == 422

    def test_case_insensitive_duplicate_chains_rejected(self, pki):
        payload = self._valid_payload(pki)
        good = payload["endpoints"][0]["candidate_chains"][0]
        payload["endpoints"][0]["candidate_chains"] = [
            good, [h.upper() for h in good]]
        r = post(payload)
        assert r.status_code == 422

    def test_uppercase_fingerprints_accepted_and_output_lowercase(self, pki):
        payload = self._valid_payload(pki)
        payload["clients"][0]["trust_roots"] = [
            r.upper() for r in payload["clients"][0]["trust_roots"]]
        payload["endpoints"][0]["candidate_chains"] = [
            [h.upper() for h in ch]
            for ch in payload["endpoints"][0]["candidate_chains"]]
        r = post(payload)
        assert r.status_code == 200, r.text
        result = r.json()["results"][0]
        for ch in result["selected_chains"]:
            assert all(h == h.lower() for h in ch)

    def test_empty_chain_rejected(self, pki):
        payload = self._valid_payload(pki)
        payload["endpoints"][0]["candidate_chains"] = [[]]
        r = post(payload)
        assert r.status_code == 422

    def test_duplicate_client_ids(self, pki):
        payload = self._valid_payload(pki)
        payload["clients"].append(dict(payload["clients"][0]))
        r = post(payload)
        assert r.status_code == 422

    def test_duplicate_relationship(self, pki):
        payload = self._valid_payload(pki)
        payload["relationships"].append(dict(payload["relationships"][0]))
        r = post(payload)
        assert r.status_code == 422

    def test_relationship_unknown_client(self, pki):
        payload = self._valid_payload(pki)
        payload["relationships"][0]["client_id"] = "ghost"
        r = post(payload)
        assert r.status_code == 422

    def test_relationship_unknown_endpoint(self, pki):
        payload = self._valid_payload(pki)
        payload["relationships"][0]["endpoint_id"] = "ghost"
        r = post(payload)
        assert r.status_code == 422

    def test_bad_oid(self, pki):
        payload = self._valid_payload(pki)
        payload["clients"][0]["acceptable_signature_oids"] = ["1.2.bad"]
        r = post(payload)
        assert r.status_code == 422

    def test_bad_key_type(self, pki):
        payload = self._valid_payload(pki)
        payload["clients"][0]["acceptable_leaf_key_types"] = ["quantum"]
        r = post(payload)
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Differential test against brute force: the HTTP answer must equal the
# exact minimum-cover optimum for every random coverage pattern.
# ---------------------------------------------------------------------------


def _brute_force_cover(chains, covers, required):
    needed = frozenset(required)
    for size in range(1, len(chains) + 1):
        for subset in itertools.combinations(chains, size):
            covered = frozenset()
            for ch in subset:
                covered |= covers[ch]
            if needed <= covered:
                return list(subset)
    return None


def test_solver_matches_bruteforce_random():
    import random
    rng = random.Random(1234)
    client_names = [f"c{i}" for i in range(6)]
    for trial in range(300):
        n_chains = rng.randint(1, 8)
        chains = tuple((f"s{j}",) for j in range(n_chains))
        covers = {
            ch: frozenset(c for c in client_names
                          if rng.random() < 0.45)
            for ch in chains
        }
        required = client_names[:rng.randint(1, 6)]
        got = minimum_variant_cover(sorted(chains), covers, required)
        want = _brute_force_cover(sorted(chains), covers, required)
        assert got == want, (trial, got, want)

