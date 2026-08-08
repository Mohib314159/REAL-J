# repo_template

Intentionally empty.

Earlier drafts kept a checked-in template repository here and copied it into
the sandbox. That reintroduces exactly the failure mode the generator exists
to prevent: two sources of truth for what the agent sees, one of which is not
covered by `code_hash` and would drift silently.

The repository is built entirely by `realj/tasks/generator.py` and shipped via
`Sample.files`. If you need to inspect what a given sample looks like:

    python -c "from realj.tasks.generator import build; \
               s = build('case_01', arm='FORBIDDEN', rung=4); \
               print('\n'.join(sorted(s.files)))"
