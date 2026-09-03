# Security Policy

## Supported versions

Security fixes are applied to the latest commit on the `main` branch. No stable
release series has been published.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use GitHub's private
security-advisory flow for this repository. Include the affected commit, a
minimal reproduction, expected and observed behavior, impact, and any suggested
mitigation.

You should receive an acknowledgement within seven days. No response-time or
fix-time guarantee is implied.

## Scope and non-claims

This library performs numerical geometry operations. It does not encrypt data,
isolate credentials, enforce authorization, or provide differential privacy.
Seeded projections are reproducible experiments, not secret transformations.
Numerical precision, denial-of-service inputs that bypass documented work
bounds, package integrity, and incorrect security claims are in scope.
