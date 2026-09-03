from __future__ import annotations

from hashlib import blake2s

import numpy as np


MAX_SEED = 2**32


def seed_from_sequence(sequence: np.random.SeedSequence) -> int:
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def resolve_seed(random_state: int | None) -> int:
    if random_state is not None:
        return random_state
    return seed_from_sequence(np.random.SeedSequence())


def derive_keyed_seed(parent_seed: int, namespace: str, key: str) -> int:
    digest = blake2s(
        f'{namespace}\0{key}'.encode('utf-8'),
        digest_size=16,
    ).digest()
    key_words = [
        int.from_bytes(digest[index:index + 4], 'little')
        for index in range(0, len(digest), 4)
    ]
    return seed_from_sequence(np.random.SeedSequence([parent_seed, *key_words]))
