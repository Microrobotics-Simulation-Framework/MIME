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

MIME is maintained by one person, so support means what he can honour:

| Version | Supported |
|---------|-----------|
| 0.3 pre-releases (`0.3.0.devN`, `0.3.0aN`) | Yes: a fix goes into the next pre-release; upgrade to it |
| 0.2.x   | Security fixes only, as the maintainer can make them |
| 0.1.x   | No |

A pre-release is not patched in place: the fix is the next one.

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

A bundle `mime pull` fetches is treated as a stranger's: it is checked before anything in it is unpacked or printed,
and a way past those checks is a vulnerability to report. Otherwise MIME assumes trusted inputs: the scripts you run
are your own, and checking what reaches MIME from elsewhere is the responsibility of the code that passes it on.
