# RFC-001: Fully Automated Nightly Rebase via Multi-Agent System

| Field        | Value                                |
|-------------|--------------------------------------|
| **Status**  | Draft                                |
| **Authors** | vLLM-Omni Team                       |
| **Created** | 2026-03-01                           |

## 1. Summary

This RFC proposes building a fully automated agent system that performs a nightly rebase of vllm-omni onto the latest upstream vLLM commit. The primary challenge is that the codebase is large enough to exceed any single LLM context window, necessitating a **multi-agent architecture** that decomposes the rebase workflow into context-bounded subtasks.

## 2. Motivation

### Current State

vllm-omni is an independent package that extends upstream vLLM (`vllm.v1.*`) with multimodal capabilities (text, image, video, audio, and diffusion models). Keeping vllm-omni in sync with upstream vLLM is critical because:

- **API drift**: vLLM frequently changes internal APIs (e.g., `GPUModelRunner`, `OutputProcessor`, `SchedulerOutput`). Each day of lag compounds merge conflict complexity.
- **Manual burden**: Past rebase efforts (e.g., `75770c93 - Rebase to vllm v0.16.0`) required significant manual work, including fixing downstream references across workers, model runners, and platform adapters.

### Why Automation Is Difficult

| Challenge | Detail |
|-----------|--------|
| **Context window limits** | vllm-omni has 1,200+ lines in `gpu_model_runner.py`, 1,500+ in `omni_stage.py`, 800+ in `gpu_generation_model_runner.py`, and dozens of model-specific files. A single-agent approach cannot hold enough context to resolve all conflicts. |
| **Semantic conflicts** | Many conflicts are not textual but semantic — upstream may rename a parameter, change a method signature, or restructure a class hierarchy. Blind merge tools fail silently. |
| **Cross-file dependencies** | A change in `vllm.v1.worker.gpu_model_runner.GPUModelRunner` can cascade into `OmniGPUModelRunner`, `GPUARModelRunner`, `GPUGenerationModelRunner`, and all platform variants (NPU, XPU, ROCm). |
| **Test validation** | After resolving conflicts, the result must be validated against CI (L1–L5 test levels, see `docs/contributing/ci/CI_5levels.md`). |

## 3. Proposed Design

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Orchestrator Agent                           │
│  (Nightly cron trigger, git operations, conflict decomposition)     │
└───────┬──────────────┬──────────────┬──────────────┬────────────────┘
        │              │              │              │
        ▼              ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Conflict    │ │  Conflict    │ │  Conflict    │ │  Validation  │
│  Resolver    │ │  Resolver    │ │  Resolver    │ │  Agent       │
│  Agent #1    │ │  Agent #2    │ │  Agent #N    │ │              │
│  (worker/)   │ │  (engine/)   │ │  (platform/) │ │  (CI runner) │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

### 3.2 Agent Roles

#### Orchestrator Agent

- **Trigger**: GitHub Actions cron schedule (e.g., `0 2 * * *` UTC)
- **Responsibilities**:
  1. Fetch the latest upstream vLLM commit (`pip install vllm` or track a pinned ref)
  2. Attempt `git rebase` or dependency version bump on a fresh branch `auto-rebase/YYYY-MM-DD`
  3. If conflicts arise, decompose them by file/module into independent work units
  4. Dispatch work units to Conflict Resolver Agents
  5. Collect results, apply patches, and trigger Validation Agent
  6. If validation passes, open a PR; if not, open an issue with diagnostics

#### Conflict Resolver Agent (per-module)

- **Context boundary**: Each agent receives only the conflicting file(s) plus their direct upstream counterpart(s) and a module-level dependency graph
- **Strategy**:
  1. Receive: `(omni_file, upstream_diff, conflict_markers, api_change_summary)`
  2. Analyze the upstream change (renamed method? new parameter? restructured class?)
  3. Apply the minimal adaptation to vllm-omni code
  4. Output: patch file + confidence score + human-review flags
- **Module partitions** (based on current codebase structure):
  - `worker/` — `OmniGPUModelRunner`, `GPUARModelRunner`, `GPUGenerationModelRunner`
  - `engine/` — `OmniInputProcessor`, `MultimodalOutputProcessor`, `OmniEngineArgs`
  - `entrypoints/` — `omni_stage.py`, `omni_llm.py`, `async_omni_llm.py`
  - `model_executor/` — Model-specific code (Qwen3, MiMo, GLM, etc.)
  - `platforms/` — NPU, XPU, ROCm adapters
  - `diffusion/` — DiffusionModelRunner and related code
  - `core/sched/` — Scheduler adaptations

#### Validation Agent

- **Responsibilities**:
  1. Run lint checks (`ruff`, `pre-commit`)
  2. Run import-level smoke tests
  3. Trigger CI L1/L2 tests (unit tests, basic e2e)
  4. Report pass/fail with logs
- **Escalation**: If validation fails, feed error context back to the relevant Conflict Resolver Agent for a retry (up to N attempts)

### 3.3 Context Management Strategy

The key innovation is **bounded context windows** per agent:

| Strategy | Description |
|----------|-------------|
| **File-level isolation** | Each Conflict Resolver Agent sees at most 2-3 files (~2-4K lines total) |
| **API summary injection** | Instead of full upstream source, agents receive a structured summary of API changes (method signatures, parameter changes, class hierarchy diffs) |
| **Dependency graph pruning** | Only relevant imports and call sites are included in context |
| **Iterative refinement** | If a patch introduces cross-module issues, the Orchestrator re-dispatches with expanded context |

### 3.4 Workflow

```
1. [Cron] Trigger nightly at 02:00 UTC
2. [Orchestrator] git fetch upstream; attempt rebase/version bump
3. [Orchestrator] If clean → skip to step 7
4. [Orchestrator] Parse conflicts → group by module → create work units
5. [Resolver Agents] Resolve conflicts in parallel (bounded context per agent)
6. [Orchestrator] Collect patches → apply sequentially → check for cross-module issues
7. [Validation Agent] Run lint + L1/L2 tests
8. [Orchestrator] If pass → open PR "Auto-rebase to vLLM <commit>"
9. [Orchestrator] If fail → retry up to 3 times with error feedback
10. [Orchestrator] If still failing → open issue with diagnostics for human review
```

### 3.5 GitHub Actions Integration

```yaml
name: Nightly Auto-Rebase
on:
  schedule:
    - cron: '0 2 * * *'
  workflow_dispatch:

jobs:
  auto-rebase:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Run Orchestrator Agent
        run: python scripts/auto_rebase/orchestrator.py
        env:
          AGENT_API_KEY: ${{ secrets.AGENT_API_KEY }}
      - name: Open PR or Issue
        run: python scripts/auto_rebase/finalize.py
```

## 4. Key Technical Decisions

### 4.1 Rebase vs. Dependency Bump

vllm-omni imports vLLM as a dependency rather than forking it. The "rebase" is actually:

1. Bumping the pinned vLLM version in `requirements/common.txt` or `setup.py`
2. Adapting vllm-omni code to API changes in the new vLLM version
3. Updating Dockerfiles (e.g., `Dockerfile.ci` base image `vllm/vllm-openai:v0.16.0`, `Dockerfile.rocm` notes)

### 4.2 Agent Backend

- Recommend using a code-specialized LLM with 128K+ context window
- Each Conflict Resolver Agent should receive structured prompts with:
  - The specific vllm-omni file(s) being adapted
  - A diff of the upstream API change
  - The inheritance/import chain relevant to the file
  - Test files that cover the changed code

### 4.3 Rollback Safety

- All changes happen on a dedicated branch (`auto-rebase/YYYY-MM-DD`)
- The PR requires at least one human approval before merge
- Failed rebases are tracked in issues with full diagnostic context

## 5. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Agent produces semantically wrong patches | Confidence scoring + human review for low-confidence patches |
| Cross-module inconsistencies | Orchestrator performs a global consistency check after collecting all patches |
| Upstream breaking changes too large | Fall back to opening a detailed issue rather than a broken PR |
| API key / cost management | Rate-limit agent calls; cache upstream API summaries across runs |
| Flaky CI tests | Distinguish test failures caused by rebase vs. pre-existing flakiness |

## 6. Success Metrics

- **Automation rate**: % of nightly rebases that produce a clean PR without human intervention
- **Time to rebase**: Wall-clock time from cron trigger to PR creation (target: < 30 minutes)
- **Conflict resolution accuracy**: % of agent-resolved conflicts that pass human review
- **CI pass rate**: % of auto-rebase PRs that pass L1/L2 CI on first attempt

## 7. Implementation Plan

| Phase | Scope | Timeline |
|-------|-------|----------|
| Phase 1 | Orchestrator skeleton: git operations, conflict detection, branch management | 2 weeks |
| Phase 2 | Conflict Resolver Agents: per-module resolution with bounded context | 3 weeks |
| Phase 3 | Validation Agent: lint + smoke test integration | 1 week |
| Phase 4 | GitHub Actions integration + PR/issue creation | 1 week |
| Phase 5 | Monitoring, confidence tuning, feedback loops | Ongoing |

## 8. Open Questions

1. Should the system attempt to rebase daily or only when upstream has significant changes?
2. What is the maximum acceptable cost per rebase run?
3. Should platform-specific adapters (NPU, XPU, ROCm) be included in the nightly rebase or handled separately?
4. How should we handle upstream changes that require new test coverage in vllm-omni?
