# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in MIME, please report it responsibly:

**Email**: nick@microrobotica.org

Please include:
- Description of the vulnerability
- Steps to reproduce
- Affected versions
- Any suggested fix (optional)

## Response Timeline

- **Acknowledgement**: within 3 business days
- **Initial assessment**: within 7 business days
- **Fix or mitigation**: best effort, prioritised by severity

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Dependency Monitoring

MIME monitors its core dependencies for known security vulnerabilities using two mechanisms:

1. **GitHub Dependabot** — enabled on the repository; automatically monitors PyPI dependencies for published CVEs and creates pull requests for security-relevant version bumps.

2. **Manual changelog review** — JAX ecosystem libraries (JAX, jaxlib) and MADDENING are reviewed at each MIME release for correctness-affecting changes (XLA compiler changes, numerical behaviour changes, MADDENING anomaly registry updates) that may not be classified as security vulnerabilities.

Security-relevant dependency updates are flagged in the `Security` section of `CHANGELOG.md`.

## Scope

MIME is a physics computation library, not a network service. The primary security concerns are:

- **Supply chain**: malicious or compromised dependencies
- **Numerical correctness**: silent computation errors (see `docs/validation/known_anomalies.yaml`)
- **Denial of service**: pathological inputs that cause excessive memory or compute usage

MIME assumes trusted inputs (see `docs/regulatory/intended_use.md`). Input sanitisation and validation is the responsibility of the downstream integration layer.
