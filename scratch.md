Proposed plan

Phase 0: Decisions needed from management (now)
1. Seat count. 50 seats are counted on active users, and there are about 390–460 Jira accounts. How many people actually need to log in? This is the main licensing question.
2. Cutover model. Either one weekend for everything, or in waves (recommended, below).
3. What happens to Jira afterwards. Options are read-only archive, kept for N months, or switched off. Jira 10.3.23 has a native instance-wide read-only mode.
4. Who approves the freeze window and who communicates it to users.

Phase 1: Build what is still missing (the real critical path)
- Delta/freeze sync. It's unbuilt. Drift is real: CRTX gained 9 comments during its own 5-hour run. It must pick up new issues, comments, worklogs, status changes and attachments since the import.
- Onboarding tooling. Unlock accounts, restore real e-mail addresses and add memberships within the seat budget. Run the locked-member mail probe on PAR first.
- Jira-writing tools. The 5 in-house tools write to Jira via agit-extras and aren't re-pointed. GNR alone has 207 k worklogs. The sXs write-back is phase-1 scope per Geert. These are the biggest unknowns, and I can't estimate them until the plugin owner is found.
- Perforce forward path. The changeset daemon has never run against Perforce or AWS, and Jira's SCM tables stop at 2026-08-18.
- Arm the dormant jobs (resolution date), and fix the 11-30 expiry of the SQL migration lane. Cutover must finish before then, or the date must be extended.
- Full dress rehearsal of the delta pass, to measure real durations.

Phase 2: Waves. Small projects first, as you suggested:

┌──────┬────────────────────┬───────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Wave │      Projects      │   Work packages   │                                                  Purpose                                                  │
├──────┼────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 1    │ SWFLX, OP, IOT     │ 138 / 120 / 370   │ Prove the cutover runbook with real users                                                                 │
├──────┼────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 2    │ BOD, PE, APC       │ 455 / 828 / 2,665 │ Both flows, more users                                                                                    │
├──────┼────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 3    │ CRTX               │ 18,862            │ First large one (~5 h run)                                                   │
├──────┼────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 4    │ AWP                │ 24,351            │ Large                                                                                                     │
├──────┼────────────────────┼───────────────────┼──────────────────────────────────────────────────────────────────────────────┤
│ 5    │ APR + ASA together │ 65,377 + 10,304   │ Same Jira project (ASA is a filter split of APR), with cross-boundary parents, so they can't be separated │
└──────┴────────────────────┴───────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────┘

Per-wave runbook
1. Announce one week ahead.
2. Do a dry-run delta the day before.
3. Freeze the project in Jira.
4. Run the final delta.
5. Run validate_db, which checks counts, hours and authors against Jira.
6. Onboard the project's users.
7. Go live.
8. Run hypercare for a few days.

Rollback: Jira stays intact, so rollback means unfreezing it. That is only clean until the first real edit is made in OpenProject.

Phase 3: Hypercare and Jira decommission.

Caveats to state honestly

- Per-project freeze in Jira is not proven. Native read-only mode is instance-wide, so it needs an agreed window. Freezing one project at a time probably means a permission-scheme change, which I haven't verified.
- Durations are unmeasured. Known figures are 12–15 h for a full APR run and 5 h for CRTX. The delta-pass time is unknown until it's built and rehearsed. I'd give Arnauld milestones and dependencies, not calendar dates yet.
- Known and accepted gaps: about 1,139 work packages with refused history (live drift), 8Jira validators or screens. Sprints and boards are not migrated.

Risks to show

Seat limit, the Jira-writing tools, the unbuilt delta, the 11-30 lane expiry, user adoption, and freeze length on APR (the longest).