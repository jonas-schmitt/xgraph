# Contributing

Start with the [glossary](GLOSSARY.md), [development guide](DEVELOPMENT.md) and
[public API](docs/api.md). The APIs are experimental. Small, reproducible changes
to the library, documentation and synthetic examples are welcome.

## Issues and pull requests

Describe the expected behavior, what happened, the source commit and environment,
and the smallest synthetic reproducer. Keep credentials, private graphs, model
weights and personal data out of issues. Report vulnerabilities through the
[security policy](SECURITY.md).

For a correctness change, first add a focused failing synthetic test, then make
the smallest fix. Keep CPU oracle behavior and GPU engine behavior consistent;
a CPU result alone cannot validate CUDA or serving behavior. Follow the existing
Python style, add type annotations where they clarify a boundary, and document
public behavior. Avoid unrelated refactoring.

A pull request should explain the problem, resulting behavior and validation
actually performed. State missing GPU or integration validation explicitly.
Keep one coherent change together with its tests and documentation. Documentation
should name existing paths and flags; put volatile measurements in separately
identified reports, with inputs, versions, configuration, seed, command and commit.

## Validation

The public source profile has a small CPU onboarding check and a documentation
build, described in [development](DEVELOPMENT.md#testing-and-validation-layers).
It also contains a curated synthetic oracle/adapter/engine test subset. Tests
that need CUDA, torch or vLLM require their own compatible environment; they are
not qualified by the CPU onboarding check. The full research checkout retains
additional gates and records outside this source profile.

## License

You retain copyright in your contributions. By submitting project-owned code,
you agree to license it under [Apache License 2.0](LICENSE). Preserve third-party
notices and identify the origin and license of any imported code. Vendored GWalk
has a separate CC BY 4.0 license; see [licensing](docs/licensing.md) and
[third-party notices](THIRD_PARTY_NOTICES.md). No copyright assignment or separate
contributor agreement is requested here.
