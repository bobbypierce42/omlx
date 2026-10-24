# CI

General static checks and release metadata jobs use `kalana-ci`.
MLX unit tests use `[self-hosted, kalana-macos-ci, macOS]`. Native Metal wheel builds
also require the automatic `ARM64` label, macOS 15 or later, and an Xcode Metal
toolchain. Before compiling, CI uses `xcrun metal` to compile AIR and link a real
Metal library, matching the checked-in CMake builds. It installs Xcode's optional
MetalToolchain component if that probe fails; a separate legacy `metallib`
executable is not required. Native wheel builds and kernel import checks still
run for every supported Python version.

Pull requests and pushes run the Python 3.11/3.12/3.13 unit suite, critical Python
lint checks, and the same native wheel build and import smoke test used by releases.
JUnit results and built wheels are uploaded as artifacts. Python environments and
wheel smoke environments live under the runner's job-specific temporary directory.
Only release events attach wheels to a release; CI calls do not publish.

External fork pull requests do not execute on the self-hosted runners. After
reviewing a fork contribution, a maintainer can copy the reviewed change to a branch
in this repository to run its checks. Do not use `pull_request_target` to execute
untrusted source on these runners.

Runner registration, labels, and repository access are managed outside this PR.
The current GitHub integration cannot list runners (HTTP 403). Existing run
39835986854 confirms all three unit-test Python versions have already succeeded
on the `kalana-macos-ci` runner family; native wheel prerequisites still need to
be satisfied on that family.
