from pathlib import Path

PROMPTS = (
    "pr_code_suggestions_prompts.toml",
    "pr_code_suggestions_prompts_not_decoupled.toml",
    "pr_code_suggestions_reflect_prompts.toml",
)


def test_code_suggestion_prompts_require_json_output():
    prompt_dir = Path(__file__).parents[2] / "pr_agent/settings/code_suggestions"

    for filename in PROMPTS:
        prompt = (prompt_dir / filename).read_text()
        assert "valid JSON" in prompt
        assert "valid YAML" not in prompt
        assert "```yaml" not in prompt
