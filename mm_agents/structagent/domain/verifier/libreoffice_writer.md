# LibreOffice Writer domain knowledge

## Verify spec shape (init_ledger)

For a libreoffice_writer outcome, emit `kind="writer_verify"` with a
structured `writer_checks` list — NOT `kind="shell_command"` with a
hand-written `python3 -c`. The framework builds the verify body
deterministically. Full schema + ops (`table_after_heading`,
`table_count`) are in the "WRITER_VERIFY" block injected into the
init_ledger prompt.

Do NOT use `a11y_match` as the verify for a document-property
outcome: a menu item being clicked does not imply the underlying
property was set — read the document via `writer_verify` instead.
