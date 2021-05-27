from normality import normalize
import fingerprints


def maybe_fingerprint(texts):
    """Generate a sequence of comparable texts for an entity. This also
    generates a `fingerprint`, i.e. a version of the name where all tokens
    are sorted alphabetically, and some parts, such as company suffixes,
    have been removed."""
    seen = set()
    for text in texts:
        plain = normalize(text, ascii=True)
        if plain is not None and plain not in seen:
            seen.add(plain)
            yield plain
        fp = fingerprints.generate(text)
        if fp is not None and len(fp) > 6 and fp not in seen:
            seen.add(fp)
            yield fp
