# Security Policy

## Reporting a vulnerability

Report privately through GitHub's
[Security tab](https://github.com/Nyvo2010/veritas/security/advisories/new)
("Report a vulnerability") rather than opening a public issue. Include the
affected version, a description, and a minimal reproduction if you have one.

You can expect an initial reply within a few days. Please give us a reasonable
window to ship a fix before disclosing publicly.

## Secrets

Veritas stores API keys in `~/.config/veritas/settings.json` (mode `0600` on
POSIX) and, optionally, in a local `.env` file. Both are gitignored. Nothing is
uploaded anywhere by Veritas itself: your keys go only to the endpoint and the
search provider you configure.

If you find a stored key in a repository, issue, or forum post, treat it as
compromised and rotate it at the provider.
