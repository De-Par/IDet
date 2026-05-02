# Doxygen API Reference

[Back to README](../README.md) | [Documentation index](index.md) | Previous: [Troubleshooting & FAQ](troubleshooting.md)

**IDet** ships a local Doxygen setup for browsing the public API, internal engine boundaries, algorithms, examples, and checked integration tests from one HTML tree.


## Requirements

Install Doxygen and Graphviz:

```bash
# Ubuntu / Debian
sudo apt-get install -y doxygen graphviz

# macOS
brew install doxygen graphviz
```

`doxygen` is required. `dot` from Graphviz is recommended because class, include, and directory graphs use it.


## Generate

```bash
scripts/generate_doxygen.sh --clean --check
```

The generated site is written to:

```text
docs/doxygen/html/index.html
```

`--check` verifies that the main HTML entry points were produced.


## View

Serve the generated HTML from the repository root:

```bash
scripts/generate_doxygen.sh --serve 8000
```

Open:

```text
http://127.0.0.1:8000/
```

For a static preview without a server, open `docs/doxygen/html/index.html` directly in a browser.


## What Is Included

| Area | Included files |
|:---|:---|
| Public C++ API | `include/idet`, `include/yuvv` |
| Library internals | `src/lib/idet`, `src/lib/yuvv` |
| Applications | `src/app` |
| Examples | `examples` |
| Tests | `tests` |
| Narrative docs | `README.md`, `docs/*.md` |

Generated HTML is intentionally ignored by git; regenerate it locally when needed.

🔝 [Back to top](#doxygen-api-reference)
