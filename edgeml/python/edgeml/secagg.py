"""Client-side Secure Aggregation (SecAgg) for federated learning.

Implements Shamir secret sharing over a finite field to mask model updates
before uploading them to the server. The server can only reconstruct the
*aggregate* of all masked updates once enough shares are collected, so
individual client updates remain private.

Protocol phases (client perspective):
  1. **Setup** -- receive session config (threshold, field size, participant list).
  2. **Share keys** -- generate a per-round random seed, split it into Shamir
     shares, and send one share to each other participant via the server.
  3. **Masked upload** -- add the mask derived from the seed to the model update
     and upload the masked update.
  4. **Unmask** -- if a peer drops out, reveal the share so the server can
     reconstruct that peer's mask and subtract it from the aggregate.

The implementation uses only the Python standard library (``secrets``, ``struct``,
``hashlib``) plus ``httpx`` (already a dependency of the SDK).
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default Mersenne prime used as finite-field modulus (2^127 - 1).
DEFAULT_FIELD_SIZE = (1 << 127) - 1

# Chunk size used when converting model bytes <-> field elements.
_CHUNK_BYTES = 4


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SecAggConfig:
    """Client-side view of a SecAgg session configuration."""
    session_id: str
    round_id: str
    threshold: int
    total_clients: int
    field_size: int = DEFAULT_FIELD_SIZE
    key_length: int = 256
    noise_scale: Optional[float] = None


@dataclass
class ShamirShare:
    """A single Shamir secret share."""
    index: int
    value: int
    modulus: int

    def to_bytes(self) -> bytes:
        mod_bytes = self.modulus.to_bytes(16, "big")
        return (
            struct.pack(">I", self.index)
            + self.value.to_bytes(16, "big")
            + struct.pack(">I", len(mod_bytes))
            + mod_bytes
        )

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> Tuple["ShamirShare", int]:
        index = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4
        value = int.from_bytes(data[offset : offset + 16], "big")
        offset += 16
        mod_len = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4
        modulus = int.from_bytes(data[offset : offset + mod_len], "big")
        offset += mod_len
        return cls(index=index, value=value, modulus=modulus), offset


# ---------------------------------------------------------------------------
# Shamir helpers (pure functions, no I/O)
# ---------------------------------------------------------------------------

def _mod_inverse(a: int, m: int) -> int:
    """Modular multiplicative inverse via extended Euclidean algorithm."""

    def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = _extended_gcd(b % a, a)
        return gcd, y1 - (b // a) * x1, x1

    gcd, x, _ = _extended_gcd(a % m, m)
    if gcd != 1:
        raise ValueError(f"Modular inverse does not exist for {a} mod {m}")
    return (x % m + m) % m


def _evaluate_polynomial(coefficients: List[int], x: int, modulus: int) -> int:
    """Evaluate polynomial at *x* using Horner's method in GF(*modulus*)."""
    result = coefficients[-1]
    for i in range(len(coefficients) - 2, -1, -1):
        result = (result * x + coefficients[i]) % modulus
    return result


def generate_shares(
    secret: int,
    threshold: int,
    total_shares: int,
    modulus: int = DEFAULT_FIELD_SIZE,
) -> List[ShamirShare]:
    """Split *secret* into *total_shares* Shamir shares.

    Any *threshold* shares are sufficient to reconstruct the secret.
    """
    if threshold < 1:
        raise ValueError("threshold must be >= 1")
    if threshold > total_shares:
        raise ValueError("threshold must be <= total_shares")

    # Build random polynomial with secret as the constant term.
    coefficients = [secret % modulus]
    for _ in range(threshold - 1):
        coefficients.append(secrets.randbelow(modulus))

    shares: List[ShamirShare] = []
    for i in range(1, total_shares + 1):
        y = _evaluate_polynomial(coefficients, i, modulus)
        shares.append(ShamirShare(index=i, value=y, modulus=modulus))
    return shares


def reconstruct_secret(shares: List[ShamirShare]) -> int:
    """Reconstruct the secret from a list of shares via Lagrange interpolation at x=0."""
    if not shares:
        raise ValueError("Need at least one share")

    modulus = shares[0].modulus
    result = 0

    for i, share_i in enumerate(shares):
        numerator = 1
        denominator = 1
        for j, share_j in enumerate(shares):
            if i != j:
                numerator = (numerator * (0 - share_j.index)) % modulus
                denominator = (denominator * (share_i.index - share_j.index)) % modulus

        lagrange_coeff = (numerator * _mod_inverse(denominator, modulus)) % modulus
        result = (result + share_i.value * lagrange_coeff) % modulus

    return result


# ---------------------------------------------------------------------------
# Field-element encoding / decoding
# ---------------------------------------------------------------------------

def model_bytes_to_field_elements(data: bytes, modulus: int = DEFAULT_FIELD_SIZE) -> List[int]:
    """Convert raw bytes into a list of finite-field elements (4-byte chunks)."""
    elements: List[int] = []
    for i in range(0, len(data), _CHUNK_BYTES):
        chunk = data[i : i + _CHUNK_BYTES]
        if len(chunk) < _CHUNK_BYTES:
            chunk = chunk + b"\x00" * (_CHUNK_BYTES - len(chunk))
        value = struct.unpack(">I", chunk)[0]
        elements.append(value % modulus)
    return elements


def field_elements_to_model_bytes(elements: List[int]) -> bytes:
    """Convert finite-field elements back to raw bytes."""
    parts: List[bytes] = []
    for elem in elements:
        parts.append(struct.pack(">I", elem % (1 << 32)))
    return b"".join(parts)


# ---------------------------------------------------------------------------
# Mask generation from a seed
# ---------------------------------------------------------------------------

def _derive_mask_elements(
    seed: bytes,
    count: int,
    modulus: int = DEFAULT_FIELD_SIZE,
) -> List[int]:
    """Derive *count* pseudorandom mask elements from *seed* using HMAC-SHA256."""
    elements: List[int] = []
    counter = 0
    while len(elements) < count:
        h = hashlib.sha256(seed + struct.pack(">I", counter)).digest()
        # Each hash gives 32 bytes -> 8 four-byte field elements.
        for off in range(0, len(h), _CHUNK_BYTES):
            if len(elements) >= count:
                break
            val = struct.unpack(">I", h[off : off + _CHUNK_BYTES])[0]
            elements.append(val % modulus)
        counter += 1
    return elements


# ---------------------------------------------------------------------------
# High-level client helper
# ---------------------------------------------------------------------------

class SecAggClient:
    """Client-side SecAgg state machine.

    Typical usage inside ``FederatedClient.participate_in_round``::

        sac = SecAggClient(config)
        shares_for_peers = sac.generate_key_shares()
        # ... send shares_for_peers to server ...
        masked_update = sac.mask_model_update(raw_update_bytes)
        # ... upload masked_update ...
    """

    def __init__(self, config: SecAggConfig) -> None:
        self.config = config
        self._seed: bytes = secrets.token_bytes(32)
        self._shares: Optional[List[ShamirShare]] = None
        self._mask_elements: Optional[List[int]] = None

    # Phase 1 -- share keys ------------------------------------------------

    def generate_key_shares(self) -> List[ShamirShare]:
        """Split this client's random seed into Shamir shares.

        Returns a list of ``total_clients`` shares. Share *i* should be
        delivered to participant *i* (1-indexed).
        """
        # Convert seed to a single big integer for sharing.
        secret_int = int.from_bytes(self._seed, "big") % self.config.field_size
        self._shares = generate_shares(
            secret=secret_int,
            threshold=self.config.threshold,
            total_shares=self.config.total_clients,
            modulus=self.config.field_size,
        )
        return list(self._shares)

    # Phase 2 -- masked upload ----------------------------------------------

    def mask_model_update(self, update_bytes: bytes) -> bytes:
        """Add a pseudorandom mask to the serialised model update.

        The mask is derived deterministically from ``self._seed`` so that the
        server can reconstruct it (via Shamir) and remove it from the aggregate.
        """
        elements = model_bytes_to_field_elements(update_bytes, self.config.field_size)
        mask = _derive_mask_elements(self._seed, len(elements), self.config.field_size)
        self._mask_elements = mask

        masked: List[int] = []
        for e, m in zip(elements, mask):
            masked.append((e + m) % self.config.field_size)

        return field_elements_to_model_bytes(masked)

    # Phase 3 -- unmask (dropout handling) ----------------------------------

    def get_seed_share_for_peer(self, peer_index: int) -> Optional[ShamirShare]:
        """Return the share destined for *peer_index* (1-based).

        Called when a peer drops out and the server requests this client's
        share of that peer's seed so it can reconstruct the dropout's mask.
        """
        if self._shares is None:
            return None
        for share in self._shares:
            if share.index == peer_index:
                return share
        return None

    # Serialisation helpers -------------------------------------------------

    @staticmethod
    def serialize_shares(shares: List[ShamirShare]) -> bytes:
        """Serialize a list of shares for network transmission."""
        buf = struct.pack(">I", len(shares))
        for s in shares:
            buf += s.to_bytes()
        return buf

    @staticmethod
    def deserialize_shares(data: bytes) -> List[ShamirShare]:
        """Deserialize shares received from the server or a peer."""
        count = struct.unpack(">I", data[:4])[0]
        offset = 4
        shares: List[ShamirShare] = []
        for _ in range(count):
            share, offset = ShamirShare.from_bytes(data, offset)
            shares.append(share)
        return shares
