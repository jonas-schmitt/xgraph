# Internals

The CPU oracle specifies legal next tokens and resolved spans. The native engine
implements the same state transitions for batched GPU decoding. Read the
contract before changing either implementation.

```{toctree}
:maxdepth: 1

architecture
engine-design
engine-contract
kg-versions-adr
```

These pages describe the library's design. They do not establish model quality,
serving compatibility or measured speed. See [performance scope](performance.md).
