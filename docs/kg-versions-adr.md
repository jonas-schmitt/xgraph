# KG versions over one compiled index

A KG version is a named, registered set of adjacency edits over a base index.
A request pins a version for its lifetime; another request can use the base or
a different registered version. Labels and identifiers remain those of the base.

Edits are grouped by subject/relation. Each group replaces every object of that
pair; a `None` object represents explicit pair deletion in the string-ID format.
Unedited pairs retain their base edges. An edit cannot introduce new labels;
unwritable additions are counted. The resulting reads must agree with an index
rebuilt under the same labels and alias policy.

The compiled version tables retain the entire adjacency of edited subjects,
including unedited pairs. Registration memory can therefore be large for hub
edits. Versions are fixed when an engine is built. Selecting an existing version
is different from ingesting edits or rebuilding that engine.

Saved edits carry their content identity and recorded base binding. Default
admission trusts the saved index identity; explicit verification checks content.
See [versions and edits](guides/kg-versions.md) for APIs and limits and
[RDF projection](guides/rdflib.md) for graph-difference conversion.
