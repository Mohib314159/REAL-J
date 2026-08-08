# Sandbox for REAL-J rollouts.
#
# Deliberately boring. The repository the agent works in is written per-sample
# by Inspect from Sample.files, so nothing task-specific is baked into the
# image -- which is what lets the same image serve every case, arm and rung
# without becoming a place where the rungs could quietly diverge.
FROM python:3.12-slim-bookworm

# pytest is the only thing the task needs, and the scorer needs it too:
# task_success.py runs the hidden tests inside this container after the
# rollout. git is present because R4 and R5 are meant to look like a checked
# out repository, and a repository with no git is a tell.
RUN pip install --no-cache-dir pytest==8.3.4 \
 && apt-get update \
 && apt-get install -y --no-install-recommends git ripgrep less \
 && rm -rf /var/lib/apt/lists/*

# Scoring directory. Created here so the scorer never has to reason about
# whether it can mkdir under whatever user the agent left things as.
RUN mkdir -p /opt/scoring /workspace/repo

WORKDIR /workspace/repo

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

CMD ["tail", "-f", "/dev/null"]
