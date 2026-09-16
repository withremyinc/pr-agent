import copy
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest

from pr_agent.servers import github_review_cycle as cycle


class FakeGitHub:
    repository = "org/repo"

    def __init__(self):
        self.pull = {
            "number": 7, "state": "open", "draft": False, "body": "", "changed_files": 2,
            "html_url": "https://github.com/org/repo/pull/7",
            "user": {"login": "author", "type": "User"},
            "head": {"sha": "head", "ref": "feature", "repo": {"full_name": self.repository}},
            "base": {"sha": "base", "ref": "main"},
        }
        self.current_threads = []
        self.checks = []
        self.reviews = []
        self.resolutions = []
        self.publications = []
        self.run = {
            "status": "completed", "conclusion": "success", "run_attempt": 1,
            "path": ".github/workflows/qodo-code-review.yml", "event": "pull_request",
            "head_sha": "head", "head_repository": {"full_name": self.repository},
        }
        self.login = "remy-ops-bot"

    def pr(self, number):
        return copy.deepcopy(self.pull)

    def unchanged(self, number, head, base):
        return cycle.GitHub.unchanged(self, number, head, base)

    def threads(self, number):
        return list(self.current_threads)

    def resolve(self, thread, resolved):
        self.current_threads = [replace(t, resolved=resolved) if t.id == thread.id else t
                                for t in self.current_threads]
        self.resolutions.append((thread.id, resolved))

    def start_check(self, head):
        id_ = len(self.checks) + 1
        self.checks.append({"id": id_, "name": cycle.CHECK_NAME, "app": {"slug": "github-actions"},
                            "status": "in_progress", "conclusion": None,
                            "external_id": "qodo-cycle-v1:123:1", "output": {}})
        return id_

    def finish_check(self, check_id, receipt, reason):
        self.checks[check_id - 1].update(
            status="completed", conclusion="success" if receipt.complete else "failure",
            output={"summary": receipt.model_dump_json(), "text": reason})

    def publish(self, number, head, finding):
        self.publications.append(finding)
        self.current_threads.append(cycle.Thread(f"thread-{len(self.current_threads)}", False,
                                                  finding.body + "\n" + finding.marker, finding.path))

    def receipt_artifact(self, run_id, number):
        return cycle.Receipt.model_validate_json(self.checks[-1]["output"]["summary"])

    def latest_review_run(self, head):
        return 123

    def trusted_workflow(self, head, base):
        return True

    def repo(self, path):
        return path

    def pages(self, path, key=None):
        if path.endswith("check-runs"):
            return iter(self.checks)
        if path.endswith("reviews"):
            return iter(self.reviews)
        raise AssertionError(path)

    def request(self, method, path, **kwargs):
        if path == "/user":
            return {"login": self.login}
        if path == "actions/runs/123":
            return self.run
        if method == "POST" and path == "pulls/7/reviews":
            self.reviews.append({"id": len(self.reviews) + 1, "state": {"COMMENT": "COMMENTED", "APPROVE": "APPROVED"}[kwargs["json"]["event"]],
                                 "user": {"login": self.login}, "commit_id": kwargs["json"]["commit_id"],
                                 "body": kwargs["json"]["body"]})
            return {}
        if method == "PUT" and path.endswith("/dismissals"):
            review_id = int(path.split("/")[-2])
            for review in self.reviews:
                if review["id"] == review_id:
                    review["state"] = "DISMISSED"
            return {}
        raise AssertionError((method, path))


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setattr(cycle, "analyze", AsyncMock(return_value=([], True, "complete")))
    monkeypatch.setattr(cycle, "recheck", AsyncMock(return_value=cycle.Recheck(verdict="open", reason="Still broken")))


def old_finding(github, resolved=False):
    finding = cycle.Finding("src.ts", 1, 1, "The old defect")
    github.current_threads.append(cycle.Thread("old", resolved, finding.body + "\n" + finding.marker, finding.path))
    return finding


async def test_complete_review_waits_for_manual_resolution_then_approves_once():
    github = FakeGitHub()
    old_finding(github)
    await cycle.review_cycle(github, 7)
    assert not cycle.approve_if_ready(github, 7)
    github.resolve(github.current_threads[0], True)
    assert cycle.approve_if_ready(github, 7)
    assert cycle.approve_if_ready(github, 7)
    assert len(github.reviews) == 1
    assert github.reviews[0]["commit_id"] == "head"


async def test_reopened_thread_revokes_only_our_automatic_approval():
    github = FakeGitHub()
    old_finding(github, resolved=True)
    await cycle.review_cycle(github, 7)
    assert cycle.reconcile_approval(github, 7)
    github.reviews.append({"id": 99, "state": "APPROVED", "user": {"login": "human"},
                           "commit_id": "head", "body": "LGTM"})
    github.resolve(github.current_threads[0], False)
    assert not cycle.reconcile_approval(github, 7)
    assert github.reviews[0]["state"] == "DISMISSED"
    assert github.reviews[1]["state"] == "APPROVED"
    github.resolve(github.current_threads[0], True)
    assert cycle.reconcile_approval(github, 7)
    assert github.reviews[-1]["state"] == "APPROVED"


async def test_explicit_fixed_verdict_resolves_and_allows_approval(monkeypatch):
    github = FakeGitHub()
    old_finding(github)
    monkeypatch.setattr(cycle, "recheck", AsyncMock(return_value=cycle.Recheck(verdict="fixed", reason="Bounds check added")))
    receipt = await cycle.review_cycle(github, 7)
    assert not github.resolutions
    assert receipt.fixed_threads == ["old"]
    assert cycle.approve_if_ready(github, 7)
    assert github.resolutions == [("old", True)]
    github.resolve(github.current_threads[0], False)
    assert not cycle.approve_if_ready(github, 7)
    assert not github.current_threads[0].resolved


@pytest.mark.parametrize("verdict", ["open", "uncertain"])
async def test_absence_from_new_findings_does_not_resolve_old_thread(monkeypatch, verdict):
    github = FakeGitHub()
    old_finding(github)
    monkeypatch.setattr(cycle, "recheck", AsyncMock(return_value=cycle.Recheck(verdict=verdict, reason="No proof of fix")))
    await cycle.review_cycle(github, 7)
    assert not github.resolutions
    assert not cycle.approve_if_ready(github, 7)


async def test_incomplete_review_cannot_resolve_or_approve_even_after_manual_resolution(monkeypatch):
    github = FakeGitHub()
    old_finding(github)
    monkeypatch.setattr(cycle, "analyze", AsyncMock(return_value=([], False, "omitted files")))
    monkeypatch.setattr(cycle, "recheck", AsyncMock(return_value=cycle.Recheck(verdict="fixed", reason="Fixed")))
    await cycle.review_cycle(github, 7)
    assert not github.resolutions
    github.resolve(github.current_threads[0], True)
    assert not cycle.approve_if_ready(github, 7)


async def test_failed_recheck_leaves_every_thread_open_and_records_failure(monkeypatch):
    github = FakeGitHub()
    old_finding(github)
    monkeypatch.setattr(cycle, "recheck", AsyncMock(side_effect=ValueError("invalid model JSON")))
    with pytest.raises(ValueError):
        await cycle.review_cycle(github, 7)
    assert not github.resolutions
    assert github.checks[-1]["conclusion"] == "failure"
    assert not cycle.approve_if_ready(github, 7)


async def test_repeated_run_deduplicates_and_redetected_finding_gets_an_open_thread(monkeypatch):
    github = FakeGitHub()
    finding = cycle.Finding("src.ts", 1, 1, "New defect")
    monkeypatch.setattr(cycle, "analyze", AsyncMock(return_value=([finding], True, "complete")))
    await cycle.review_cycle(github, 7)
    await cycle.review_cycle(github, 7)
    assert len(github.publications) == 1
    github.resolve(github.current_threads[0], True)
    await cycle.review_cycle(github, 7)
    assert github.current_threads[0].resolved
    assert not github.current_threads[1].resolved
    assert len(github.publications) == 2


async def test_new_push_during_resolution_reopens_what_this_run_resolved(monkeypatch):
    github = FakeGitHub()
    old_finding(github)
    monkeypatch.setattr(cycle, "recheck", AsyncMock(return_value=cycle.Recheck(verdict="fixed", reason="Fixed")))
    resolve = github.resolve

    def race(thread, resolved):
        resolve(thread, resolved)
        if resolved:
            github.pull["head"]["sha"] = "new-head"

    await cycle.review_cycle(github, 7)
    github.resolve = race
    with pytest.raises(RuntimeError, match="changed"):
        cycle.approve_if_ready(github, 7)
    assert not github.current_threads[0].resolved
    assert not cycle.approve_if_ready(github, 7)


async def test_unpublished_finding_blocks_completion(monkeypatch):
    github = FakeGitHub()
    finding = cycle.Finding("src.ts", 1, 1, "Defect")
    monkeypatch.setattr(cycle, "analyze", AsyncMock(return_value=([finding], True, "complete")))
    github.publish = lambda *args: None
    with pytest.raises(RuntimeError, match="Not all findings"):
        await cycle.review_cycle(github, 7)
    assert not cycle.approve_if_ready(github, 7)


@pytest.mark.parametrize("change", ["head", "base", "failed-run", "attempt", "workflow", "fork", "draft",
                                    "author", "missing-receipt", "forged-app", "latest-incomplete", "non-Qodo-thread"])
async def test_approval_fails_closed(change):
    github = FakeGitHub()
    await cycle.review_cycle(github, 7)
    if change in {"head", "base"}:
        github.pull[change]["sha"] = "new"
    elif change == "failed-run":
        github.run["conclusion"] = "failure"
    elif change == "attempt":
        github.run["run_attempt"] = 2
    elif change == "workflow":
        github.run["path"] = ".github/workflows/other.yml"
    elif change == "fork":
        github.pull["head"]["repo"]["full_name"] = "other/repo"
    elif change == "draft":
        github.pull["draft"] = True
    elif change == "author":
        github.pull["user"]["type"] = "Bot"
    elif change == "missing-receipt":
        github.checks[-1]["output"]["summary"] = "{}"
    elif change == "forged-app":
        github.checks[-1]["app"]["slug"] = "other-app"
    elif change == "latest-incomplete":
        github.start_check("head")
    else:
        # threads() already filters ownership. A returned open thread must block approval.
        old_finding(github)
    assert not cycle.approve_if_ready(github, 7)
    assert not github.reviews


async def test_wrong_approval_identity_never_writes():
    github = FakeGitHub()
    await cycle.review_cycle(github, 7)
    github.login = "someone-else"
    with pytest.raises(ValueError, match="Approval token"):
        cycle.approve_if_ready(github, 7)
    assert not github.reviews


def test_thread_discovery_paginates_and_requires_marker_and_author(monkeypatch):
    github = cycle.GitHub("org/repo", "not-a-real-token")
    body = old_finding(FakeGitHub()).marker

    def node(id_, author, text):
        return {"id": id_, "isResolved": False, "path": "src.ts",
                "comments": {"nodes": [{"body": text, "author": {
                    "login": author, "__typename": "Bot" if author == "github-actions[bot]" else "User"}}]}}

    pages = [
        {"nodes": [node("human", "author", body), node("unmarked", "github-actions[bot]", "hello")],
         "pageInfo": {"hasNextPage": True, "endCursor": "next"}},
        {"nodes": [node("owned", "github-actions[bot]", body),
                   node("legacy", "github-actions[bot]", "<!-- pr-agent-dedup: abcdef123456 -->")],
         "pageInfo": {"hasNextPage": False, "endCursor": None}},
    ]
    graphql = Mock(side_effect=[{"repository": {"pullRequest": {"reviewThreads": page}}} for page in pages])
    monkeypatch.setattr(github, "graphql", graphql)
    assert [thread.id for thread in github.threads(7)] == ["owned", "legacy"]
    assert graphql.call_args.args[1]["cursor"] == "next"


@pytest.mark.parametrize("kind,login,owned", [
    ("Bot", "github-actions", True),
    ("Bot", "github-actions[bot]", True),
    ("User", "github-actions", False),
    ("Bot", "other-reviewer", False),
])
def test_graphql_bot_identity_matches_rest_without_trusting_user_lookalikes(monkeypatch, kind, login, owned):
    github = cycle.GitHub("org/repo", "not-a-real-token")
    body = cycle.Finding("src.ts", 1, 1, "Defect").marker
    data = {"repository": {"pullRequest": {"reviewThreads": {
        "pageInfo": {"hasNextPage": False, "endCursor": None},
        "nodes": [{"id": "thread", "isResolved": False, "path": "src.ts", "comments": {"nodes": [
            {"body": body, "author": {"__typename": kind, "login": login}}
        ]}}],
    }}}}
    monkeypatch.setattr(github, "graphql", lambda *args: data)
    assert bool(github.threads(7)) == owned


def test_github_graphql_errors_are_not_an_empty_thread_list(monkeypatch):
    github = cycle.GitHub("org/repo", "not-a-real-token")
    monkeypatch.setattr(github, "request", lambda *args, **kwargs: {"errors": [{"message": "denied"}]})
    with pytest.raises(RuntimeError, match="GraphQL"):
        github.threads(7)


async def test_missing_file_is_uncertain_and_makes_no_model_call(monkeypatch):
    github = FakeGitHub()
    old_finding(github)
    github.file = lambda *args: None
    # Use the real recheck, not the autouse fixture's model boundary.
    # Load the function captured before monkeypatching from the module-level reference below.
    result = await REAL_RECHECK(github, github.current_threads[0], "head")
    assert result.verdict == "uncertain"


REAL_RECHECK = cycle.recheck
REAL_ANALYZE = cycle.analyze


async def test_receipt_in_check_cannot_override_immutable_artifact():
    github = FakeGitHub()
    await cycle.review_cycle(github, 7)
    receipt = github.receipt_artifact(123, 7).model_copy(update={"complete": False})
    github.receipt_artifact = lambda *args: receipt
    assert not cycle.approve_if_ready(github, 7)
    assert not github.reviews


async def test_newer_review_run_invalidates_an_older_success_receipt():
    github = FakeGitHub()
    await cycle.review_cycle(github, 7)
    github.latest_review_run = lambda head: 124
    assert not cycle.approve_if_ready(github, 7)
    assert not github.reviews


async def test_workflow_edits_require_human_review():
    github = FakeGitHub()
    await cycle.review_cycle(github, 7)
    github.trusted_workflow = lambda *args: False
    assert not cycle.approve_if_ready(github, 7)
    assert not github.reviews


async def test_old_event_cannot_review_a_newer_commit():
    github = FakeGitHub()
    with pytest.raises(RuntimeError, match="triggering"):
        await cycle.review_cycle(github, 7, "old-head")
    assert not github.checks


@pytest.mark.parametrize("defect", [None, "review-omission", "suggestion-omission", "review-chunk",
                                     "suggestion-chunk", "parse", "security", "finding-limit",
                                     "suggestion-limit"])
async def test_analysis_completeness_uses_both_passes_and_limits(monkeypatch, defect):
    from types import SimpleNamespace

    data = {"review": {"key_issues_to_review": [], "security_concerns": "No"}}
    reviewer = SimpleNamespace(
        run=AsyncMock(), prediction="valid", prediction_data=data,
        remaining_files_list=[], review_failed_chunk_count=0,
        _validate_review_schema=lambda data: True,
    )
    suggestions = SimpleNamespace(
        run=AsyncMock(), data={"code_suggestions": []}, remaining_files_list=[],
        failed_chunk_count=0, parse_failure_count=0, prediction_list=[{"code_suggestions": []}],
    )
    if defect == "review-omission":
        reviewer.remaining_files_list = ["unreviewed.ts"]
    elif defect == "suggestion-omission":
        suggestions.remaining_files_list = ["unreviewed.ts"]
    elif defect == "review-chunk":
        reviewer.review_failed_chunk_count = 1
    elif defect == "suggestion-chunk":
        suggestions.failed_chunk_count = 1
    elif defect == "parse":
        suggestions.parse_failure_count = 1
    elif defect == "security":
        data["review"]["security_concerns"] = "Unsafe SQL interpolation"
    elif defect == "finding-limit":
        data["review"]["key_issues_to_review"] = [
            {"relevant_file": "src.ts", "start_line": 1, "end_line": 1,
             "issue_header": "Bug", "issue_content": "Details"}
        ] * cycle.get_settings().pr_reviewer.num_max_findings
    elif defect == "suggestion-limit":
        suggestions.prediction_list = [{"code_suggestions": [None]
                                       * cycle.get_settings().pr_code_suggestions.num_code_suggestions_per_chunk}]
    monkeypatch.setattr(cycle, "PRReviewer", lambda *args: reviewer)
    monkeypatch.setattr(cycle, "PRCodeSuggestions", lambda *args: suggestions)
    previous = cycle.get_settings().config.publish_output
    _, complete, details = await REAL_ANALYZE("https://github.com/org/repo/pull/7")
    assert complete == (defect is None)
    assert cycle.get_settings().config.publish_output == previous
    assert "security_concerns" in details


@pytest.mark.parametrize("blocked_by", ["workflow", "artifact", "identity"])
async def test_fixed_receipt_cannot_resolve_before_trust_and_identity_checks(monkeypatch, blocked_by):
    github = FakeGitHub()
    old_finding(github)
    monkeypatch.setattr(cycle, "recheck", AsyncMock(return_value=cycle.Recheck(verdict="fixed", reason="Fixed")))
    await cycle.review_cycle(github, 7)
    if blocked_by == "workflow":
        github.trusted_workflow = lambda *args: False
    elif blocked_by == "artifact":
        receipt = github.receipt_artifact(123, 7).model_copy(update={"fixed_threads": []})
        github.receipt_artifact = lambda *args: receipt
    else:
        github.login = "wrong-user"
    if blocked_by == "identity":
        with pytest.raises(ValueError):
            cycle.approve_if_ready(github, 7)
    else:
        assert not cycle.approve_if_ready(github, 7)
    assert not github.resolutions
    assert not github.reviews


async def test_analysis_exception_restores_publication_settings(monkeypatch):
    from types import SimpleNamespace

    reviewer = SimpleNamespace(run=AsyncMock(side_effect=RuntimeError("provider failed")))
    monkeypatch.setattr(cycle, "PRReviewer", lambda *args: reviewer)
    previous = cycle.get_settings().config.publish_output
    with pytest.raises(RuntimeError):
        await REAL_ANALYZE("https://github.com/org/repo/pull/7")
    assert cycle.get_settings().config.publish_output == previous
