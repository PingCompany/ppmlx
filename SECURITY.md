# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| latest  | Yes       |

## Reporting a Vulnerability

**Do not open a public issue for security vulnerabilities.**

Email **rafal@ppmlx.dev** with:

- Description of the vulnerability
- Steps to reproduce
- Potential impact

### Response Timeline

- **48 hours** — acknowledgment of your report
- **7 days** — initial assessment and severity rating
- **30 days** — fix or mitigation for confirmed vulnerabilities

## Scope

This policy covers the ppmlx CLI and API server code. Issues in upstream dependencies (MLX, mlx-lm, mlx-vlm, HuggingFace models) should be reported to their respective maintainers.

## Design Note

ppmlx is designed as a local-first tool. The server binds to `127.0.0.1` by default. Exposing it on `0.0.0.0` is explicitly opt-in and documented as a user responsibility.
