# Security policy

## Scope

This repository is an educational defensive-security prototype. The included
dataset is synthetic, and the classifier is not a replacement for enterprise
email-security controls.

When working with real email data, treat message bodies, headers, URLs, and
attachments as untrusted input. Do not commit credentials, tokens, private
datasets, or sensitive email content.

## Reporting a vulnerability

Please do not disclose exploitable details in a public issue. Use GitHub's
private vulnerability reporting if it is enabled for this repository; otherwise
contact the repository maintainer privately through the GitHub profile for
`tracolerd`.

Include the affected file or component, reproduction steps, impact, and any
safe mitigation you have identified. Allow time for assessment before public
disclosure.

## Supported maintenance baseline

Changes should pass the repository's CI checks, including tests, Ruff, and
dependency auditing. Security fixes must preserve the documented CLI contract
unless a breaking change is explicitly documented.
