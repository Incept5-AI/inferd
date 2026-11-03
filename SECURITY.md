# Security policy

## Status and supported versions

inferd is experimental software. Only the latest commit on `main` receives fixes; there are no supported release
branches yet.

## Trust model

inferd is built for a **trusted local user** on a single Mac:

- The HTTP front end binds to loopback by default. It has no multi-tenant isolation: any local process that can
  reach the port, or the engine's Unix socket, can create, read and delete contexts and sessions.
- Binding to another interface (`--host`) requires an API key (`INFERD_API_KEY`), but there is still no per-owner
  isolation of contexts, sessions or statistics. Do not expose it to untrusted networks.
- Model checkpoints are treated as untrusted input and validated as they load. Load only checkpoints you have reason
  to trust, and report any checkpoint that crashes the loader.
- Image URLs and local image files are disabled by default. Enabling them (`INFERD_IMAGE_URLS=1`,
  `INFERD_IMAGE_FILES=1`) lets requests make the server fetch URLs, including internal hosts, or read files. Do this
  only for trusted clients.
- The Neural Engine prefill split uses private Apple APIs and may stop working with an OS update. It is off unless
  `ANE_UNITS` is set.

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's private vulnerability reporting for this repository (Security →
Report a vulnerability). Include the commit, the machine and
macOS version, and a reproduction. We aim to acknowledge reports within five working days.
