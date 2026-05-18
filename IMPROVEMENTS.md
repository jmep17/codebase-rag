# Improvements

This is the working improvement tracker for codebase-rag. Tasks are marked done
as soon as the implementation or decision is complete.

## Now

- [x] Create this improvement tracker.
- [x] Add a `doctor` command for local environment checks.
- [ ] Add focused tests for `chat.agent_turn` confirmation and error events.
- [ ] Add WebSocket transport tests for auth, confirmations, cancellation, and disconnects.

## Next

- [ ] Build a local retrieval eval harness with fixture repos, golden questions,
      expected paths, and recall metrics.
- [ ] Make normal `index` remove stale chunks for deleted or renamed files.
- [ ] Replace tuple-based agent events with typed event objects.
- [ ] Add permission profiles for observe/edit/shell/web sessions.
- [ ] Tighten audit redaction for secrets, tokens, environment-like values, and
      command output.
- [ ] Improve CLI help grouping and actionable setup errors.
- [ ] Add dev tooling and CI for tests, linting, import-light checks, and CLI
      smoke tests.
