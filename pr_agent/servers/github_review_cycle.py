"""GitHub-only review lifecycle. A complete analysis is not an approval."""

import asyncio
import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass
from typing import Literal

import requests
from pydantic import BaseModel, ConfigDict, Field

from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler
from pr_agent.algo.inline_comment_dedup import BODY_MARKER_RE
from pr_agent.algo.token_handler import TokenHandler
from pr_agent.config_loader import get_settings
from pr_agent.log import get_logger
from pr_agent.tools.pr_code_suggestions import PRCodeSuggestions
from pr_agent.tools.pr_reviewer import PRReviewer

CHECK_NAME = "Qodo review coverage"
APPROVAL_MARKER = "<!-- qodo-approval:v1 -->"
MARKER_PREFIX = "<!-- qodo-finding:v1:"


class Receipt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal[1] = 1
    head: str
    base: str
    run_id: int
    attempt: int
    complete: bool
    fixed_threads: list[str] = Field(default_factory=list)


class Recheck(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    verdict: Literal["fixed", "open", "uncertain"]
    reason: str = Field(min_length=1)


@dataclass(frozen=True)
class Finding:
    path: str
    start: int
    end: int
    body: str

    @property
    def marker(self):
        # Ignore positions, which move as unrelated lines change.
        digest = hashlib.sha256((self.path + "\n" + self.body).encode()).hexdigest()
        return f"{MARKER_PREFIX}{digest} -->"


@dataclass(frozen=True)
class Thread:
    id: str
    resolved: bool
    body: str
    path: str


def marker(body):
    for line in reversed(body.splitlines()):
        if line.startswith(MARKER_PREFIX) and line.endswith(" -->"):
            digest = line[len(MARKER_PREFIX):-4]
            if len(digest) == 64 and all(c in "0123456789abcdef" for c in digest):
                return line
    legacy = BODY_MARKER_RE.search(body)
    return legacy.group(0) if legacy else None


def eligible(pr, repository):
    return (pr["state"] == "open" and not pr["draft"]
            and pr["head"]["repo"] is not None
            and pr["head"]["repo"]["full_name"] == repository
            and pr["user"]["type"] != "Bot"
            and pr["base"]["ref"] != "release/prod"
            and not pr["head"]["ref"].startswith("mq-bot-")
            and "# Aviator metadata" not in (pr["body"] or ""))


class GitHub:
    def __init__(self, repository, token):
        self.repository = repository
        self.base_url = os.getenv("GITHUB_API_URL", "https://api.github.com").rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}",
                                     "Accept": "application/vnd.github+json",
                                     "X-GitHub-Api-Version": "2022-11-28"})
        self.author = get_settings().github_review_cycle.comment_author

    def request(self, method, path, **kwargs):
        response = self.session.request(method, self.base_url + path, timeout=30, **kwargs)
        response.raise_for_status()
        return response.json()

    def repo(self, path):
        return f"/repos/{self.repository}/{path}"

    def pages(self, path, key=None):
        page = 1
        while True:
            data = self.request("GET", self.repo(path), params={"per_page": 100, "page": page})
            items = data[key] if key else data
            yield from items
            if len(items) < 100:
                return
            page += 1

    def pr(self, number):
        return self.request("GET", self.repo(f"pulls/{number}"))

    def unchanged(self, number, head, base):
        pr = self.pr(number)
        return eligible(pr, self.repository) and pr["head"]["sha"] == head and pr["base"]["sha"] == base

    def graphql(self, query, variables):
        data = self.request("POST", "/graphql", json={"query": query, "variables": variables})
        if data.get("errors"):
            messages = "; ".join(error["message"] for error in data["errors"])
            raise RuntimeError(f"GitHub GraphQL request failed: {messages}")
        return data["data"]

    def threads(self, number):
        owner, name = self.repository.split("/")
        cursor = None
        result = []
        while True:
            data = self.graphql("""
                query($owner:String!, $name:String!, $number:Int!, $cursor:String) {
                  repository(owner:$owner, name:$name) {
                    pullRequest(number:$number) {
                      reviewThreads(first:100, after:$cursor) {
                        pageInfo { hasNextPage endCursor }
                        nodes { id isResolved path comments(first:1) { nodes { body author { __typename login } } } }
                      }
                    }
                  }
                }
            """, {"owner": owner, "name": name, "number": number, "cursor": cursor})
            connection = data["repository"]["pullRequest"]["reviewThreads"]
            for node in connection["nodes"]:
                root = node["comments"]["nodes"][0]
                author = root["author"]
                if not author:
                    continue
                login = author["login"]
                # GraphQL Bot.login omits REST's reserved [bot] suffix.
                if author["__typename"] == "Bot" and not login.endswith("[bot]"):
                    login += "[bot]"
                if login == self.author and marker(root["body"]):
                    result.append(Thread(node["id"], node["isResolved"], root["body"], node["path"]))
            if not connection["pageInfo"]["hasNextPage"]:
                return result
            cursor = connection["pageInfo"]["endCursor"]
            if not cursor:
                raise RuntimeError("Missing GitHub thread pagination cursor")

    def resolve(self, thread, resolved):
        operation = "resolveReviewThread" if resolved else "unresolveReviewThread"
        data = self.graphql(f"""
            mutation($id:ID!) {{ {operation}(input:{{threadId:$id}}) {{ thread {{ isResolved }} }} }}
        """, {"id": thread.id})
        if data[operation]["thread"]["isResolved"] != resolved:
            raise RuntimeError("GitHub did not update the thread resolution")

    def file(self, path, head):
        import base64
        from urllib.parse import quote

        try:
            data = self.request("GET", self.repo(f"contents/{quote(path, safe='/')}"), params={"ref": head})
        except requests.HTTPError as error:
            if error.response.status_code == 404:
                return None  # A deleted/moved file is not evidence that a finding was fixed.
            raise
        if data["type"] != "file" or data["encoding"] != "base64":
            return None
        return base64.b64decode(data["content"]).decode("utf-8")

    def start_check(self, head):
        return self.request("POST", self.repo("check-runs"), json={
            "name": CHECK_NAME, "head_sha": head, "status": "in_progress",
            "external_id": f"qodo-cycle-v1:{os.environ['GITHUB_RUN_ID']}:{os.environ['GITHUB_RUN_ATTEMPT']}",
        })["id"]

    def finish_check(self, check_id, receipt, reason):
        self.request("PATCH", self.repo(f"check-runs/{check_id}"), json={
            "status": "completed", "conclusion": "success" if receipt.complete else "failure",
            "output": {"title": "Review complete" if receipt.complete else "Review incomplete",
                       "summary": receipt.model_dump_json(), "text": reason[:60000]},
        })

    def receipt_artifact(self, run_id, number):
        artifacts = list(self.pages(f"actions/runs/{run_id}/artifacts", "artifacts"))
        artifacts = [item for item in artifacts
                     if item["name"] == f"qodo-review-receipt-{number}" and not item["expired"]]
        if len(artifacts) != 1:
            raise ValueError("Missing or ambiguous review receipt artifact")
        path = self.repo(f"actions/artifacts/{artifacts[0]['id']}/zip")
        response = self.session.get(self.base_url + path, timeout=30)
        response.raise_for_status()
        if len(response.content) > 1024 * 1024:
            raise ValueError("Review receipt archive is too large")
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            if archive.namelist() != ["qodo-review-receipt.json"]:
                raise ValueError("Unexpected review receipt archive contents")
            if archive.getinfo("qodo-review-receipt.json").file_size > 65536:
                raise ValueError("Review receipt is too large")
            return Receipt.model_validate_json(archive.read("qodo-review-receipt.json"))

    def latest_review_run(self, head):
        from urllib.parse import quote

        workflow = quote(get_settings().github_review_cycle.review_workflow.rsplit("/", 1)[-1], safe="")
        runs = self.request("GET", self.repo(f"actions/workflows/{workflow}/runs"),
                            params={"head_sha": head, "event": "pull_request", "per_page": 1})["workflow_runs"]
        return runs[0]["id"] if runs else None

    def trusted_workflow(self, head, base):
        path = get_settings().github_review_cycle.review_workflow
        at_head = self.request("GET", self.repo(f"contents/{path}"), params={"ref": head})
        at_base = self.request("GET", self.repo(f"contents/{path}"), params={"ref": base})
        return at_head["sha"] == at_base["sha"]

    def publish(self, number, head, finding):
        comment = {"path": finding.path, "line": finding.end, "side": "RIGHT",
                   "body": f"{finding.body}\n\n{finding.marker}"}
        if finding.start < finding.end:
            comment.update(start_line=finding.start, start_side="RIGHT")
        # commit_id binds this publication to the analyzed commit, not a newer push.
        self.request("POST", self.repo(f"pulls/{number}/reviews"), json={
            "commit_id": head, "event": "COMMENT", "comments": [comment],
        })


async def analyze(pr_url):
    """Reuse the existing analysis tools; this cycle alone owns publication and resolution."""
    settings = get_settings()
    publish, propagate = settings.config.publish_output, settings.config.propagate_tool_errors
    try:
        settings.set("config.publish_output", False)
        settings.set("config.propagate_tool_errors", True)
        reviewer = PRReviewer(pr_url)
        await reviewer.run()
        if not reviewer.prediction:
            raise ValueError("No review prediction")
        data = reviewer.prediction_data or reviewer._load_valid_review_yaml(reviewer.prediction)
        if not reviewer._validate_review_schema(data):
            raise ValueError("Invalid review schema")
        suggestions = PRCodeSuggestions(pr_url)
        await suggestions.run()
        if suggestions.data is None:
            raise ValueError("No suggestions analysis")
    finally:
        settings.set("config.publish_output", publish)
        settings.set("config.propagate_tool_errors", propagate)

    findings = []
    for issue in data["review"]["key_issues_to_review"]:
        findings.append(Finding(issue["relevant_file"].strip(), int(issue["start_line"]), int(issue["end_line"]),
                                f"**Qodo review: {issue['issue_header'].strip()}**\n\n{issue['issue_content'].strip()}"))
    for suggestion in suggestions.data["code_suggestions"]:
        # Plain review comments, not automatically applicable patches. The original code and proposed
        # replacement remain available to the reviewer without offering unvalidated edits to GitHub.
        body = f"**Qodo suggestion**\n\n{suggestion['suggestion_content'].strip()}"
        if suggestion.get("improved_code"):
            body += "\n\nProposed code:\n````\n" + suggestion["improved_code"].strip() + "\n````"
        findings.append(Finding(suggestion["relevant_file"].strip(), int(suggestion["relevant_lines_start"]),
                                int(suggestion["relevant_lines_end"]), body))

    complete = (not reviewer.remaining_files_list and not reviewer.review_failed_chunk_count
                and not suggestions.remaining_files_list and not suggestions.failed_chunk_count
                and not suggestions.parse_failure_count
                and len(data["review"]["key_issues_to_review"]) < settings.pr_reviewer.num_max_findings
                and str(data["review"].get("security_concerns", "")).strip().lower() == "no"
                and settings.pr_code_suggestions.max_suggestions_per_file == 0
                and all(len(prediction["code_suggestions"]) < settings.pr_code_suggestions.num_code_suggestions_per_chunk
                        for prediction in suggestions.prediction_list))
    complete = complete and settings.pr_code_suggestions.suggestions_score_threshold == 0
    details = json.dumps({
        "review_omitted_files": reviewer.remaining_files_list,
        "suggestions_omitted_files": suggestions.remaining_files_list,
        "failed_review_chunks": reviewer.review_failed_chunk_count,
        "failed_suggestion_chunks": suggestions.failed_chunk_count,
        "security_concerns": data["review"].get("security_concerns"),
    })
    return findings, complete, details


async def recheck(github, thread, head):
    code = github.file(thread.path, head)
    if code is None:
        return Recheck(verdict="uncertain", reason="The original file is missing or unavailable")
    system = get_settings().github_review_cycle.recheck_instructions
    user = json.dumps({"finding": thread.body, "path": thread.path, "current_file": code})
    if TokenHandler().count_tokens(system + user) > get_settings().github_review_cycle.recheck_max_tokens:
        return Recheck(verdict="uncertain", reason="The complete file exceeds the recheck budget")
    response, _ = await LiteLLMAIHandler().chat_completion(
        model=get_settings().config.model, system=system, user=user, temperature=0.0)
    return Recheck.model_validate_json(response)


async def review_cycle(github, number, expected_head=None):
    pr = github.pr(number)
    if expected_head is not None and pr["head"]["sha"] != expected_head:
        raise RuntimeError("The triggering PR commit is no longer current")
    if not eligible(pr, github.repository):
        return
    head, base = pr["head"]["sha"], pr["base"]["sha"]
    receipt = Receipt(head=head, base=base, run_id=int(os.environ["GITHUB_RUN_ID"]),
                      attempt=int(os.environ["GITHUB_RUN_ATTEMPT"]), complete=False)
    check_id = github.start_check(head)
    reason = "Review did not complete"
    try:
        before = github.threads(number)
        findings, complete, details = await analyze(pr["html_url"])
        if pr["changed_files"] > 3000:
            complete = False
            details += "\nGitHub's 3,000-file diff limit prevents complete coverage."
        by_marker = {finding.marker: finding for finding in findings}
        old_open = [thread for thread in before if not thread.resolved and marker(thread.body) not in by_marker]
        if len(old_open) > get_settings().github_review_cycle.max_rechecks:
            complete = False
            details += "\nOutstanding findings exceed the recheck limit."
            verdicts = []
        else:
            # Bound concurrent model calls. Do not act on any verdict until every recheck has returned.
            semaphore = asyncio.Semaphore(4)

            async def check(thread):
                async with semaphore:
                    return thread, await recheck(github, thread, head)

            verdicts = await asyncio.gather(*(check(thread) for thread in old_open))
        if not github.unchanged(number, head, base):
            raise RuntimeError("PR changed during analysis")
        current = github.threads(number)
        existing_open = {marker(thread.body) for thread in current if not thread.resolved}
        for fingerprint, finding in by_marker.items():
            if not github.unchanged(number, head, base):
                raise RuntimeError("PR changed during publication")
            if fingerprint not in existing_open:
                # Actions cannot change thread resolution. Redetected findings get a new open thread.
                github.publish(number, head, finding)
        published = {marker(thread.body) for thread in github.threads(number) if not thread.resolved}
        if not by_marker.keys() <= published:
            raise RuntimeError("Not all findings were published inline")
        if not github.unchanged(number, head, base):
            raise RuntimeError("PR changed before completion")
        receipt.complete = complete
        if complete:
            receipt.fixed_threads = [thread.id for thread, verdict in verdicts if verdict.verdict == "fixed"]
        reason = ("All findings published. Unresolved threads block approval."
                  if complete else "Coverage, parsing, a finding limit, or security concerns require another review.")
        reason += "\n\n" + details
    finally:
        github.finish_check(check_id, receipt, reason)
    return receipt


def resolve_verified_threads(github, number, receipt, approver):
    if not receipt.fixed_threads:
        return
    applied_marker = f"<!-- qodo-resolutions:v1:{receipt.run_id}:{receipt.attempt} -->"
    reviews = list(github.pages(f"pulls/{number}/reviews"))
    if any(review["user"]["login"] == approver and applied_marker in review.get("body", "") for review in reviews):
        return  # A later manual reopen must not be undone by replaying the same receipt.
    resolved = []
    try:
        for thread in github.threads(number):
            if thread.id not in receipt.fixed_threads or thread.resolved:
                continue
            if (not github.unchanged(number, receipt.head, receipt.base)
                    or github.latest_review_run(receipt.head) != receipt.run_id):
                raise RuntimeError("PR changed before thread resolution")
            github.resolve(thread, True)
            resolved.append(thread)
        if not github.unchanged(number, receipt.head, receipt.base):
            raise RuntimeError("PR changed during thread resolution")
        github.request("POST", github.repo(f"pulls/{number}/reviews"), json={
            "event": "COMMENT", "commit_id": receipt.head,
            "body": f"Qodo applied the confirmed-fix decisions from review run {receipt.run_id}.\n\n{applied_marker}",
        })
    except Exception:
        for thread in resolved:
            github.resolve(thread, False)
        raise


def approve_if_ready(github, number, *, publish=True):
    pr = github.pr(number)
    if not eligible(pr, github.repository):
        return False
    head, base = pr["head"]["sha"], pr["base"]["sha"]
    checks = [check for check in github.pages(f"commits/{head}/check-runs", "check_runs")
              if check["name"] == CHECK_NAME and check["app"]["slug"] == "github-actions"]
    if not checks:
        return False
    check = max(checks, key=lambda item: item["id"])
    if check["status"] != "completed" or check["conclusion"] != "success":
        return False
    try:
        receipt = Receipt.model_validate_json(check["output"]["summary"])
    except ValueError:
        return False
    if (not receipt.complete or (receipt.head, receipt.base) != (head, base)
            or check["external_id"] != f"qodo-cycle-v1:{receipt.run_id}:{receipt.attempt}"):
        return False
    run = github.request("GET", github.repo(f"actions/runs/{receipt.run_id}"))
    if (github.latest_review_run(head) != receipt.run_id
            or run["status"] != "completed" or run["conclusion"] != "success"
            or run["run_attempt"] != receipt.attempt
            or run["path"] != get_settings().github_review_cycle.review_workflow
            or run["event"] != "pull_request" or run["head_sha"] != head
            or run["head_repository"]["full_name"] != github.repository):
        return False
    # Checks are writable by other workflows. The run-scoped immutable artifact is authoritative.
    try:
        artifact = github.receipt_artifact(receipt.run_id, number)
    except (ValueError, zipfile.BadZipFile):
        return False
    if artifact != receipt or not github.trusted_workflow(head, base):
        return False
    approver = github.request("GET", "/user")["login"]
    if approver != get_settings().github_review_cycle.approver or approver == pr["user"]["login"]:
        raise ValueError("Approval token must belong to the configured bot, not the PR author")
    if publish:
        resolve_verified_threads(github, number, receipt, approver)
    if any(not thread.resolved for thread in github.threads(number)):
        return False
    reviews = list(github.pages(f"pulls/{number}/reviews"))
    previous = [review for review in reviews if review["user"]["login"] == approver
                and review["commit_id"] == head and review["state"] != "COMMENTED"]
    if previous:
        state = max(previous, key=lambda item: item["id"])["state"]
        if state == "APPROVED":
            return True
        if state == "CHANGES_REQUESTED":
            return False
    if (not github.unchanged(number, head, base) or any(not t.resolved for t in github.threads(number))
            or github.latest_review_run(head) != receipt.run_id):
        return False
    if not publish:
        return True
    github.request("POST", github.repo(f"pulls/{number}/reviews"), json={
        "event": "APPROVE", "commit_id": head,
        "body": "Qodo completed a full review of this commit. All Qodo findings are resolved.\n\n" + APPROVAL_MARKER,
    })
    return True


def revoke_approval(github, number):
    approver = get_settings().github_review_cycle.approver
    if github.request("GET", "/user")["login"] != approver:
        raise ValueError("Approval token does not belong to the configured bot")
    for review in github.pages(f"pulls/{number}/reviews"):
        if (review["user"]["login"] == approver and review["state"] == "APPROVED"
                and APPROVAL_MARKER in review.get("body", "")):
            github.request("PUT", github.repo(f"pulls/{number}/reviews/{review['id']}/dismissals"), json={
                "message": "Qodo approval requirements are no longer satisfied. A fresh complete review and resolved threads are required.",
            })


def reconcile_approval(github, number):
    if not approve_if_ready(github, number):
        revoke_approval(github, number)
        return False
    # GitHub cannot atomically bind an approval to thread state. Recheck after publication too;
    # later sweeps revoke our approval if a thread is reopened or the reviewed commit changes.
    if not approve_if_ready(github, number, publish=False):
        revoke_approval(github, number)
        return False
    return True


async def run(mode, payload):
    github = GitHub(os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_TOKEN"])
    receipt = None
    if mode == "review":
        if os.environ["GITHUB_EVENT_NAME"] != "pull_request":
            raise ValueError("Review mode requires a pull_request event")
        receipt = await review_cycle(github, payload["number"], payload["pull_request"]["head"]["sha"])
    elif mode == "approve":
        if os.environ["GITHUB_EVENT_NAME"] not in {"workflow_run", "schedule", "workflow_dispatch"}:
            raise ValueError("Approval must run in a trusted workflow")
        for pr in github.pages("pulls"):
            try:
                reconcile_approval(github, pr["number"])
            except Exception:
                get_logger().exception(f"Qodo approval failed for PR #{pr['number']}")
                raise
    else:
        raise ValueError(f"Unknown GitHub review cycle mode: {mode}")
    if os.getenv("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write("review_cycle_version=1\n")
            if receipt is not None:
                output.write(f"review_cycle_receipt={receipt.model_dump_json()}\n")
