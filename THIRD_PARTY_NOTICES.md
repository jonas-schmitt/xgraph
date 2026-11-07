# Third-party notices

The root [Apache 2.0 license](LICENSE) covers project-owned code. It does not
replace the licenses below. Preserve source notices and modification markers.

## GWalk code and prompts — CC BY 4.0

`src/xgraph/gwalk_upstream/` contains adapted code and prompt text from
[MQuAKE-Remastered](https://github.com/henryzhongsc/MQuAKE-Remastered/tree/349dcc50460c251ee09a6804aab444508a016485),
revision `349dcc50460c251ee09a6804aab444508a016485`.
Creators: Shaochen Zhong, Yifan Lu, Lize Shao, Bhargav Bhushanam, Xiaocong Du,
Yixin Wan, Yucheng Shi, Daochen Zha, Yiwei Wang, Ninghao Liu, Kaixiong Zhou,
Shuai Xu, Kai-Wei Chang, Louis Feng, Vipin Chaudhary and Xia Hu.
The upstream paper is *MQuAKE-Remastered: Multi-Hop Knowledge Editing Can Only
Be Advanced with Reliable Evaluations*, ICLR 2025.

This component remains under [CC BY 4.0](LICENSES/CC-BY-4.0.txt).
The [retained source notice](src/xgraph/gwalk_upstream/__init__.py) maps upstream
files and describes xgraph's changes; changed code is marked `xgraph:`.
The authors do not endorse xgraph. The original upstream license notice is
retained in [LICENSES/GWalk-NOTICE.txt](LICENSES/GWalk-NOTICE.txt).

## ReFactX prompt text — Apache 2.0

`REFACTX_SYSTEM` and associated examples in `src/xgraph/prompts.py` adapt the
ReFactX protocol by Pozzi et al., ISWC 2025, from
[rpo19/ReFactX](https://github.com/rpo19/ReFactX/tree/ISWC2025).
The source comment describes changes: omission of entity descriptions, examples
rewritten for entity answers and the graph used, fact-span syntax and a common
answer format. The upstream Apache license is retained in
[LICENSES/ReFactX-Apache-2.0.txt](LICENSES/ReFactX-Apache-2.0.txt).
Upstream authors do not endorse this adaptation.

## Native build dependencies

CUDA builds statically incorporate nanobind. The build dependency is pinned to
`nanobind==2.12.0`; its BSD-3-Clause notice is in
[LICENSES/nanobind-BSD-3-Clause.txt](LICENSES/nanobind-BSD-3-Clause.txt).
Copyright 2022 Wenzel Jakob. The corresponding upstream source is
[wjakob/nanobind](https://github.com/wjakob/nanobind/tree/v2.12.0).

That revision uses robin-map revision
`4ec1bf19c6a96125ea22062f38c2cf5b958e448e`, copyright 2017
Thibaut Goetghebuer-Planchon. Its MIT notice is in
[LICENSES/robin-map-MIT.txt](LICENSES/robin-map-MIT.txt).
These notices accompany CPU source installs too; a CPU build contains no CUDA
extension. A separately distributed native wheel must retain these notices and
describe its additional BSD-3-Clause and MIT components in artifact metadata.

## Dependencies and research assets

Installed Python dependencies retain their own licenses; they are not vendored
by the source export. Development and research tools have separate terms and
are not all Apache-licensed. Model weights, tokenizers, downloaded datasets,
evaluation outputs and private research history are excluded from the curated
source export. The project license grants no redistribution rights to them.

Agent skills and third-party benchmark fixtures in the private research
checkout retain their local notices and are omitted from that export.
