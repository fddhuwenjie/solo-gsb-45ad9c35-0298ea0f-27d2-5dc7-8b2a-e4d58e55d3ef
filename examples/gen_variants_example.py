#!/usr/bin/env python3
"""Regenerate examples/variants_request.json from freshly built PKI material.

The example demonstrates selecting the minimum certificate-variant set for
two endpoints whose clients are heterogeneous:

    Legacy Root (RSA) -> api-rsa leaf (RSA, sha256WithRSAEncryption)
                       -> api-rsa-384 leaf (RSA, sha384WithRSAEncryption,
                          rejected: no client accepts that OID)
    Modern Root (EC)  -> Modern Intermediate (EC) -> api-ec leaf (EC)
                       -> grpc-ec leaf (EC)
    Modern Root (EC)  -> grpc-ed leaf (Ed25519; rejected: untrusted root
                          chain for the EC-only client)

Clients:
    pos-terminal : trusts Legacy Root, RSA leaves, sha256WithRSAEncryption only
    mobile-app   : trusts both roots, RSA + EC leaves, RSA/ECDSA SHA-256 OIDs
    iot-sensor   : trusts Modern Root, EC leaves, ecdsa-with-SHA256 only

For api.example.com the RSA chain covers pos-terminal + mobile-app and the EC
chain covers iot-sensor + mobile-app: the exact optimum is 2 variants (a
per-client greedy walk would also reach 2 here, but other patterns do not;
the service always solves set cover exactly). The SHA-384 chain is
cryptographically valid but selected by nobody, so it never inflates the
deployment set. grpc.example.com needs just the EC chain.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa
from cryptography.x509.oid import NameOID

NOT_BEFORE = datetime(2024, 1, 1, tzinfo=timezone.utc)
NOT_AFTER = datetime(2035, 1, 1, tzinfo=timezone.utc)

SHA256_RSA = "1.2.840.113549.1.1.11"
SHA384_RSA = "1.2.840.113549.1.1.12"
ECDSA_SHA256 = "1.2.840.10045.4.3.2"
ED25519_OID = "1.3.101.112"


def dn(cn: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def issue(subject_cn, issuer_name, issuer_key, subject_key, *,
          hash_alg=hashes.SHA256(), is_ca, sans=()):
    builder = (
        x509.CertificateBuilder()
        .subject_name(dn(subject_cn))
        .issuer_name(issuer_name)
        .public_key(subject_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(NOT_BEFORE)
        .not_valid_after(NOT_AFTER)
        .add_extension(x509.BasicConstraints(ca=is_ca, path_length=None),
                       critical=True)
    )
    if sans:
        builder = builder.add_extension(
            x509.SubjectAlternativeName([x509.DNSName(n) for n in sans]),
            critical=False,
        )
    # PureEdDSA keys sign without a hash parameter (the key fixes the OID).
    if isinstance(issuer_key, ed25519.Ed25519PrivateKey):
        return builder.sign(issuer_key, None)
    return builder.sign(issuer_key, hash_alg)


def pem(cert: x509.Certificate) -> str:
    return cert.public_bytes(serialization.Encoding.PEM).decode()


def fingerprint(cert: x509.Certificate) -> str:
    return hashlib.sha256(
        cert.public_bytes(serialization.Encoding.DER)
    ).hexdigest()


def chain(*certs) -> list[str]:
    return [fingerprint(c) for c in certs]


def main() -> None:
    # --- Legacy PKI (RSA) ---
    legacy_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    legacy_root = issue("Legacy Root CA", dn("Legacy Root CA"),
                        legacy_key, legacy_key, is_ca=True)
    api_rsa = issue(
        "api.example.com", dn("Legacy Root CA"), legacy_key,
        rsa.generate_private_key(65537, 2048), is_ca=False,
        sans=["api.example.com"],
    )
    api_rsa_384 = issue(
        "api.example.com", dn("Legacy Root CA"), legacy_key,
        rsa.generate_private_key(65537, 2048), is_ca=False,
        hash_alg=hashes.SHA384(), sans=["api.example.com"],
    )

    # --- Modern PKI (EC intermediates, one Ed25519 dead-end chain) ---
    modern_key = ec.generate_private_key(ec.SECP256R1())
    modern_root = issue("Modern Root CA", dn("Modern Root CA"),
                        modern_key, modern_key, is_ca=True)
    inter_key = ec.generate_private_key(ec.SECP256R1())
    modern_inter = issue("Modern Intermediate CA", dn("Modern Root CA"),
                         modern_key, inter_key, is_ca=True)
    api_ec = issue(
        "api.example.com", dn("Modern Intermediate CA"), inter_key,
        ec.generate_private_key(ec.SECP256R1()), is_ca=False,
        sans=["api.example.com"],
    )
    grpc_ec = issue(
        "grpc.example.com", dn("Modern Intermediate CA"), inter_key,
        ec.generate_private_key(ec.SECP256R1()), is_ca=False,
        sans=["grpc.example.com"],
    )
    ed_key = ed25519.Ed25519PrivateKey.generate()
    # A real Ed25519 root whose fingerprint none of the clients trusts.
    ed_root = issue("Ed Root CA", dn("Ed Root CA"), ed_key, ed_key, is_ca=True)
    grpc_ed = issue(
        "grpc.example.com", dn("Ed Root CA"), ed_key,
        ed25519.Ed25519PrivateKey.generate(), is_ca=False,
        sans=["grpc.example.com"],
    )

    bundle = [
        pem(api_ec), pem(api_rsa), pem(legacy_root), pem(modern_inter),
        pem(modern_root), pem(grpc_ec), pem(api_rsa_384), pem(grpc_ed),
        pem(ed_root),
    ]

    payload = {
        "pem_certificates": bundle,
        "clients": [
            {
                "client_id": "pos-terminal",
                "trust_roots": [fingerprint(legacy_root)],
                "acceptable_leaf_key_types": ["rsa"],
                "acceptable_signature_oids": [SHA256_RSA],
            },
            {
                "client_id": "mobile-app",
                "trust_roots": [fingerprint(legacy_root),
                                fingerprint(modern_root)],
                "acceptable_leaf_key_types": ["rsa", "ec"],
                "acceptable_signature_oids": [SHA256_RSA, ECDSA_SHA256],
            },
            {
                "client_id": "iot-sensor",
                "trust_roots": [fingerprint(modern_root)],
                "acceptable_leaf_key_types": ["ec"],
                "acceptable_signature_oids": [ECDSA_SHA256],
            },
        ],
        "endpoints": [
            {
                "endpoint_id": "api.example.com",
                "candidate_chains": [
                    chain(api_rsa, legacy_root),
                    chain(api_rsa_384, legacy_root),
                    chain(api_ec, modern_inter, modern_root),
                ],
            },
            {
                "endpoint_id": "grpc.example.com",
                "candidate_chains": [
                    chain(grpc_ec, modern_inter, modern_root),
                    chain(grpc_ed, ed_root),
                ],
            },
        ],
        "relationships": [
            {"client_id": "pos-terminal", "endpoint_id": "api.example.com"},
            {"client_id": "mobile-app", "endpoint_id": "api.example.com"},
            {"client_id": "iot-sensor", "endpoint_id": "api.example.com"},
            {"client_id": "iot-sensor", "endpoint_id": "grpc.example.com"},
        ],
    }

    out = Path(__file__).with_name("variants_request.json")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    print(f"  legacy root: {fingerprint(legacy_root)}")
    print(f"  modern root: {fingerprint(modern_root)}")


if __name__ == "__main__":
    main()
