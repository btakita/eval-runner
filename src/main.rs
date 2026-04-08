use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;

#[derive(Parser)]
#[command(name = "eval-runner", about = "Run LLM evals with YAML test cases and LLM-as-judge scoring")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Parser)]
enum Command {
    /// Run eval cases from a YAML file
    Run {
        /// Path to cases YAML file
        cases: PathBuf,

        /// Model to use for generating responses
        #[arg(long, default_value = "sonnet")]
        model: String,

        /// Model to use for LLM-as-judge grading
        #[arg(long, default_value = "sonnet")]
        judge_model: String,

        /// Score threshold for pass/fail (default: 0.7)
        #[arg(long, default_value = "0.7")]
        threshold: f64,

        /// Output JSON results instead of human-readable
        #[arg(long)]
        json: bool,

        /// Baseline results file for regression comparison
        #[arg(long)]
        baseline: Option<PathBuf>,

        /// System prompt file to prepend to each case
        #[arg(long)]
        system_prompt: Option<PathBuf>,

        /// Backend to use for LLM calls
        #[arg(long, default_value = "claude")]
        backend: Backend,

        /// Number of cases to run in parallel (default: 1)
        #[arg(long, default_value = "1")]
        parallel: usize,

        /// Temperature for judge model (0.0 = deterministic). Only works with anthropic-api backend.
        #[arg(long)]
        judge_temperature: Option<f64>,

        /// Filter cases by tag (e.g. --tag smoke)
        #[arg(long)]
        tag: Option<String>,

        /// Base URL for OpenAI-compatible API (default: https://api.openai.com/v1)
        #[arg(long, default_value = "https://api.openai.com/v1")]
        base_url: String,
    },
}

#[derive(Debug, Clone, ValueEnum)]
enum Backend {
    /// Claude Code CLI (uses Max/OAuth credentials)
    Claude,
    /// Anthropic API (requires ANTHROPIC_API_KEY)
    AnthropicApi,
    /// OpenAI-compatible API (requires OPENAI_API_KEY, works with GPT, Gemini, Groq, ollama, etc.)
    OpenAi,
}

// --- YAML case format ---

#[derive(Debug, Deserialize)]
struct CasesFile {
    cases: Vec<EvalCase>,
}

#[derive(Debug, Deserialize)]
struct EvalCase {
    id: String,
    description: String,
    // LLM-rubric fields
    #[serde(default)]
    diff: String,
    #[serde(default)]
    document_context: String,
    #[serde(default)]
    expected_behavior: Vec<String>,
    #[serde(default = "default_grading")]
    grading: String,
    #[allow(dead_code)]
    #[serde(default)]
    notes: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    // Latency eval fields
    #[serde(default)]
    document: Option<String>,
    #[serde(default)]
    input: Option<String>,
    #[serde(default)]
    threshold_ms: Option<u64>,
    // Budget constraints (optional, fail case if exceeded)
    #[serde(default)]
    max_input_tokens: Option<u64>,
    #[serde(default)]
    max_output_tokens: Option<u64>,
    #[serde(default)]
    max_latency_ms: Option<u64>,
}

fn default_grading() -> String {
    "llm-rubric".to_string()
}

/// Strip comments from a unified diff, matching production behavior.
///
/// In production, `diff::strip_comments` runs on both snapshot and document
/// before diff computation. Eval diffs are pre-computed from raw documents,
/// so they may contain comment lines that production would never show the LLM.
///
/// This function reconstructs both sides of the diff, strips comments from
/// each, and recomputes the unified diff. Returns `None` if no changes remain
/// after stripping (the diff was entirely comments).
fn strip_diff_comments(diff_text: &str) -> Option<String> {
    let mut snapshot_lines: Vec<String> = Vec::new();
    let mut document_lines: Vec<String> = Vec::new();

    for line in diff_text.lines() {
        // Skip unified diff headers
        if line.starts_with("--- ") || line.starts_with("+++ ") || line.starts_with("@@ ") {
            continue;
        }
        if let Some(content) = line.strip_prefix('+') {
            document_lines.push(content.to_string());
        } else if let Some(content) = line.strip_prefix('-') {
            snapshot_lines.push(content.to_string());
        } else if let Some(content) = line.strip_prefix(' ') {
            snapshot_lines.push(content.to_string());
            document_lines.push(content.to_string());
        } else {
            // Bare context line (no prefix)
            snapshot_lines.push(line.to_string());
            document_lines.push(line.to_string());
        }
    }

    let snapshot_text = snapshot_lines.join("\n") + "\n";
    let document_text = document_lines.join("\n") + "\n";

    let snapshot_stripped = agent_doc::component::strip_comments(&snapshot_text);
    let document_stripped = agent_doc::component::strip_comments(&document_text);

    if snapshot_stripped == document_stripped {
        return None; // No changes after stripping
    }

    let diff = similar::TextDiff::from_lines(&snapshot_stripped, &document_stripped);
    let has_changes = diff
        .iter_all_changes()
        .any(|c| c.tag() != similar::ChangeTag::Equal);

    if !has_changes {
        return None;
    }

    Some(
        diff.unified_diff()
            .context_radius(5)
            .header("snapshot", "document")
            .to_string(),
    )
}

// --- Results format ---

#[derive(Debug, Serialize, Deserialize)]
struct RunResults {
    timestamp: String,
    model: String,
    judge_model: String,
    threshold: f64,
    cases: Vec<CaseResult>,
    summary: Summary,
}

#[derive(Debug, Serialize, Deserialize)]
struct CaseResult {
    id: String,
    description: String,
    passed: bool,
    score: f64,
    reasoning: String,
    response_preview: String,
    latency_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_tokens: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Summary {
    total: usize,
    passed: usize,
    failed: usize,
    pass_rate: f64,
    mean_score: f64,
    total_latency_ms: u64,
    p50_latency_ms: u64,
    p95_latency_ms: u64,
    regressions: Vec<String>,
}

// --- LLM response ---

struct LlmResponse {
    text: String,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
}

// --- Backend implementations ---

async fn call_llm(
    backend: &Backend,
    model: &str,
    system: Option<&str>,
    prompt: &str,
    temperature: Option<f64>,
    base_url: &str,
) -> Result<LlmResponse> {
    match backend {
        Backend::Claude => call_claude_cli(model, system, prompt).await,
        Backend::AnthropicApi => call_anthropic_api(model, system, prompt, temperature).await,
        Backend::OpenAi => call_openai_api(model, system, prompt, temperature, base_url).await,
    }
}

async fn call_claude_cli(model: &str, system: Option<&str>, prompt: &str) -> Result<LlmResponse> {
    let mut cmd = tokio::process::Command::new("claude");
    cmd.args(["--print", "--model", model]);
    cmd.arg("--output-format").arg("json");

    if let Some(sys) = system {
        cmd.arg("--system-prompt").arg(sys);
    }

    cmd.arg(prompt);

    let output = cmd
        .output()
        .await
        .context("Failed to run `claude` CLI. Is it installed and in PATH?")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("claude CLI error (exit {}): {stderr}", output.status);
    }

    let stdout = String::from_utf8(output.stdout)
        .context("Invalid UTF-8 in claude CLI output")?;

    // Parse JSON output to extract text and usage
    let json: serde_json::Value = serde_json::from_str(&stdout)
        .context("Failed to parse claude CLI JSON output")?;

    let text = json["result"].as_str().unwrap_or("").to_string();

    let usage = &json["usage"];
    let input_tokens = if usage.is_object() {
        let input = usage["input_tokens"].as_u64().unwrap_or(0);
        let cache_creation = usage["cache_creation_input_tokens"].as_u64().unwrap_or(0);
        let cache_read = usage["cache_read_input_tokens"].as_u64().unwrap_or(0);
        let total = input + cache_creation + cache_read;
        if total > 0 {
            if cache_read > 0 || cache_creation > 0 {
                eprintln!(
                    "    [cache] read={} created={} uncached={}",
                    cache_read, cache_creation, input
                );
            }
            Some(total)
        } else {
            None
        }
    } else {
        None
    };
    let output_tokens = usage["output_tokens"].as_u64();

    Ok(LlmResponse { text, input_tokens, output_tokens })
}

async fn call_anthropic_api(model: &str, system: Option<&str>, prompt: &str, temperature: Option<f64>) -> Result<LlmResponse> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .context("ANTHROPIC_API_KEY required for anthropic-api backend")?;

    #[derive(Serialize)]
    struct ApiRequest {
        model: String,
        max_tokens: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        system: Option<Vec<SystemBlock>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        temperature: Option<f64>,
        messages: Vec<ApiMessage>,
    }

    #[derive(Serialize)]
    struct SystemBlock {
        #[serde(rename = "type")]
        block_type: String,
        text: String,
        cache_control: CacheControl,
    }

    #[derive(Serialize)]
    struct CacheControl {
        #[serde(rename = "type")]
        cache_type: String,
    }

    #[derive(Serialize)]
    struct ApiMessage {
        role: String,
        content: String,
    }

    #[derive(Deserialize)]
    struct ApiResponse {
        content: Vec<ContentBlock>,
        #[serde(default)]
        usage: Option<Usage>,
    }

    #[derive(Deserialize)]
    struct ContentBlock {
        text: String,
    }

    #[derive(Deserialize)]
    struct Usage {
        #[serde(default)]
        input_tokens: u64,
        #[serde(default)]
        output_tokens: u64,
        #[serde(default)]
        cache_creation_input_tokens: u64,
        #[serde(default)]
        cache_read_input_tokens: u64,
    }

    let system_blocks = system.map(|s| vec![SystemBlock {
        block_type: "text".to_string(),
        text: s.to_string(),
        cache_control: CacheControl {
            cache_type: "ephemeral".to_string(),
        },
    }]);

    let request = ApiRequest {
        model: model.to_string(),
        max_tokens: 4096,
        system: system_blocks,
        temperature,
        messages: vec![ApiMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }],
    };

    let client = reqwest::Client::new();
    let resp = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", &api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&request)
        .send()
        .await
        .context("Failed to send request to Anthropic API")?;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Anthropic API error {status}: {body}");
    }

    let api_resp: ApiResponse = resp.json().await.context("Failed to parse API response")?;
    let text = api_resp
        .content
        .first()
        .map(|c| c.text.clone())
        .unwrap_or_default();

    let (input_tokens, output_tokens) = match &api_resp.usage {
        Some(u) => {
            if u.cache_read_input_tokens > 0 || u.cache_creation_input_tokens > 0 {
                eprintln!(
                    "    [cache] read={} created={} uncached={}",
                    u.cache_read_input_tokens, u.cache_creation_input_tokens, u.input_tokens
                );
            }
            (Some(u.input_tokens + u.cache_read_input_tokens + u.cache_creation_input_tokens), Some(u.output_tokens))
        }
        None => (None, None),
    };

    Ok(LlmResponse { text, input_tokens, output_tokens })
}

async fn call_openai_api(
    model: &str,
    system: Option<&str>,
    prompt: &str,
    temperature: Option<f64>,
    base_url: &str,
) -> Result<LlmResponse> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .context("OPENAI_API_KEY required for open-ai backend")?;

    #[derive(Serialize)]
    struct ChatRequest {
        model: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        temperature: Option<f64>,
        messages: Vec<ChatMessage>,
    }

    #[derive(Serialize)]
    struct ChatMessage {
        role: String,
        content: String,
    }

    #[derive(Deserialize)]
    struct ChatResponse {
        choices: Vec<ChatChoice>,
    }

    #[derive(Deserialize)]
    struct ChatChoice {
        message: ChatChoiceMessage,
    }

    #[derive(Deserialize)]
    struct ChatChoiceMessage {
        content: String,
    }

    let mut messages = Vec::new();
    if let Some(sys) = system {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: sys.to_string(),
        });
    }
    messages.push(ChatMessage {
        role: "user".to_string(),
        content: prompt.to_string(),
    });

    let request = ChatRequest {
        model: model.to_string(),
        temperature,
        messages,
    };

    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));

    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {api_key}"))
        .header("content-type", "application/json")
        .json(&request)
        .send()
        .await
        .with_context(|| format!("Failed to send request to {url}"))?;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("OpenAI API error {status}: {body}");
    }

    let chat_resp: ChatResponse = resp.json().await.context("Failed to parse OpenAI API response")?;
    let text = chat_resp
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .unwrap_or_default();

    Ok(LlmResponse { text, input_tokens: None, output_tokens: None })
}

// --- Judge output ---

#[derive(Debug, Deserialize)]
struct JudgeOutput {
    score: f64,
    reasoning: String,
}

// --- Latency eval ---

async fn run_latency_case(case: &EvalCase) -> Result<CaseResult> {
    let document = case
        .document
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("latency case '{}' missing `document` field", case.id))?;
    let input = case
        .input
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("latency case '{}' missing `input` field", case.id))?;
    let threshold_ms = case.threshold_ms.ok_or_else(|| {
        anyhow::anyhow!("latency case '{}' missing `threshold_ms` field", case.id)
    })?;

    // Write the test document to a temp file
    let tmp_dir = std::env::temp_dir();
    let tmp_path = tmp_dir.join(format!("eval-runner-{}-{}.md", case.id, std::process::id()));
    std::fs::write(&tmp_path, document)
        .with_context(|| format!("Failed to write temp document for case '{}'", case.id))?;

    let start = Instant::now();

    let mut cmd = tokio::process::Command::new("agent-doc");
    cmd.arg("write")
        .arg("--force-disk")
        .arg(&tmp_path)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());

    let mut child = cmd
        .spawn()
        .context("Failed to spawn `agent-doc write`. Is agent-doc in PATH?")?;

    if let Some(mut stdin) = child.stdin.take() {
        use tokio::io::AsyncWriteExt;
        stdin
            .write_all(input.as_bytes())
            .await
            .context("Failed to write input to agent-doc stdin")?;
    }

    let output = child
        .wait_with_output()
        .await
        .context("Failed to wait for agent-doc write")?;

    let elapsed_ms = start.elapsed().as_millis() as u64;

    // Clean up temp file
    let _ = std::fs::remove_file(&tmp_path);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "agent-doc write failed (exit {}): {stderr}",
            output.status
        );
    }

    let passed = elapsed_ms < threshold_ms;
    let score = if passed {
        1.0
    } else {
        // Partial score: how close were we? 0.0 = 2x threshold or worse
        let ratio = threshold_ms as f64 / elapsed_ms as f64;
        ratio.max(0.0)
    };

    Ok(CaseResult {
        id: case.id.clone(),
        description: case.description.clone(),
        passed,
        score,
        reasoning: format!(
            "elapsed={elapsed_ms}ms threshold={threshold_ms}ms — {}",
            if passed { "PASS" } else { "EXCEEDS THRESHOLD" }
        ),
        response_preview: format!("write latency: {elapsed_ms}ms"),
        latency_ms: elapsed_ms,
        input_tokens: None,
        output_tokens: None,
    })
}

// --- LLM-rubric eval ---

async fn run_case(
    backend: &Backend,
    case: &EvalCase,
    model: &str,
    judge_model: &str,
    system_prompt: Option<&str>,
    threshold: f64,
    judge_temperature: Option<f64>,
    base_url: &str,
) -> Result<CaseResult> {
    let start = Instant::now();

    // Step 1: Generate response from the model
    // Strip comments from the diff to match production behavior.
    // In production, preflight strips comments before computing the diff.
    // `None` means the diff was entirely comments — no meaningful changes.
    let processed_diff = if case.diff.is_empty() {
        Some(case.diff.clone())
    } else {
        strip_diff_comments(&case.diff)
    };
    let diff_for_prompt = processed_diff.as_deref().unwrap_or("(no changes after comment stripping)");

    let prompt = format!(
        "You are an AI agent responding to user edits in a markdown document session.\n\n\
         **Diff (user's changes):**\n```\n{}\n```\n\n\
         **Document context:**\n{}\n\n\
         Respond to the user's changes naturally. Address what they changed.",
        diff_for_prompt, case.document_context
    );

    let response = call_llm(backend, model, system_prompt, &prompt, None, base_url).await?;

    // Step 2: Judge the response
    let expected = case
        .expected_behavior
        .iter()
        .enumerate()
        .map(|(i, b)| format!("{}. {}", i + 1, b))
        .collect::<Vec<_>>()
        .join("\n");

    let judge_prompt = format!(
        "You are an eval judge. Rate how well the following response matches the expected behavior.\n\n\
         **Case:** {}\n**Description:** {}\n\n\
         **Expected behavior:**\n{}\n\n\
         **Actual response:**\n{}\n\n\
         Rate on a scale of 0.0 to 1.0 where:\n\
         - 1.0 = perfectly matches all expected behaviors\n\
         - 0.5 = partially matches\n\
         - 0.0 = completely wrong\n\n\
         Respond with ONLY valid JSON (no markdown fences):\n\
         {{\"score\": <float>, \"reasoning\": \"<explanation>\"}}",
        case.id, case.description, expected, response.text
    );

    let judge_resp = call_llm(backend, judge_model, None, &judge_prompt, judge_temperature, base_url).await?;

    let judge_output: JudgeOutput = serde_json::from_str(&judge_resp.text)
        .unwrap_or_else(|_| {
            let cleaned = judge_resp.text
                .trim_start_matches("```json")
                .trim_start_matches("```")
                .trim_end_matches("```")
                .trim();
            serde_json::from_str(cleaned).unwrap_or(JudgeOutput {
                score: 0.0,
                reasoning: format!("Failed to parse judge response: {}", judge_resp.text),
            })
        });

    let latency_ms = start.elapsed().as_millis() as u64;

    // Check budget constraints
    let mut budget_violations = Vec::new();
    if let (Some(max), Some(actual)) = (case.max_input_tokens, response.input_tokens) {
        if actual > max {
            budget_violations.push(format!("input_tokens={actual} > max={max}"));
        }
    }
    if let (Some(max), Some(actual)) = (case.max_output_tokens, response.output_tokens) {
        if actual > max {
            budget_violations.push(format!("output_tokens={actual} > max={max}"));
        }
    }
    if let Some(max) = case.max_latency_ms {
        if latency_ms > max {
            budget_violations.push(format!("latency={latency_ms}ms > max={max}ms"));
        }
    }

    let passed = judge_output.score >= threshold && budget_violations.is_empty();
    let reasoning = if budget_violations.is_empty() {
        judge_output.reasoning
    } else {
        format!("{} | BUDGET EXCEEDED: {}", judge_output.reasoning, budget_violations.join(", "))
    };

    Ok(CaseResult {
        id: case.id.clone(),
        description: case.description.clone(),
        passed,
        score: judge_output.score,
        reasoning,
        response_preview: response.text.chars().take(200).collect(),
        latency_ms,
        input_tokens: response.input_tokens,
        output_tokens: response.output_tokens,
    })
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn print_result(result: &CaseResult) {
    let status = if result.passed {
        "\x1b[32mPASS\x1b[0m"
    } else {
        "\x1b[31mFAIL\x1b[0m"
    };
    let score = format!("{:.2}", result.score);
    let latency = format!("{:.1}s", result.latency_ms as f64 / 1000.0);

    eprintln!(
        "  [{status}] {:<30} score={score}  latency={latency}",
        result.id
    );
    if !result.passed {
        eprintln!(
            "         \x1b[33m{}\x1b[0m",
            result.reasoning.chars().take(120).collect::<String>()
        );
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Run {
            cases,
            model,
            judge_model,
            threshold,
            json,
            baseline,
            system_prompt,
            backend,
            parallel,
            judge_temperature,
            tag,
            base_url,
        } => {
            // Load cases
            let content =
                std::fs::read_to_string(&cases).context("Failed to read cases file")?;
            let cases_file: CasesFile =
                serde_yaml::from_str(&content).context("Failed to parse cases YAML")?;

            // Filter by tag if specified
            let cases_vec: Vec<EvalCase> = if let Some(ref tag_filter) = tag {
                cases_file.cases.into_iter()
                    .filter(|c| c.tags.iter().any(|t| t == tag_filter))
                    .collect()
            } else {
                cases_file.cases
            };

            // Load system prompt if provided
            let sys_prompt = system_prompt
                .map(|p| std::fs::read_to_string(p).context("Failed to read system prompt"))
                .transpose()?;

            let case_count = cases_vec.len();
            if case_count == 0 {
                anyhow::bail!("No cases found{}", tag.as_ref().map(|t| format!(" with tag '{t}'")).unwrap_or_default());
            }
            let backend_name = format!("{backend:?}").to_lowercase();
            let concurrency = parallel.max(1);

            if !json {
                eprintln!(
                    "\nRunning {case_count} cases (backend={backend_name}, model={model}, judge={judge_model}, threshold={threshold}, parallel={concurrency})...\n"
                );
            }

            let semaphore = Arc::new(Semaphore::new(concurrency));
            let backend = Arc::new(backend);
            let model = Arc::new(model);
            let judge_model = Arc::new(judge_model);
            let sys_prompt = Arc::new(sys_prompt);
            let base_url = Arc::new(base_url);

            let mut handles = Vec::new();
            for (i, case) in cases_vec.into_iter().enumerate() {
                let sem = semaphore.clone();
                let backend = backend.clone();
                let model = model.clone();
                let judge_model = judge_model.clone();
                let sys_prompt = sys_prompt.clone();
                let base_url = base_url.clone();

                let handle = tokio::spawn(async move {
                    let _permit = sem.acquire().await.unwrap();
                    eprintln!("  [{}/{}] {}...", i + 1, case_count, case.id);
                    let result = if case.grading == "latency" {
                        run_latency_case(&case).await
                    } else {
                        run_case(
                            &backend,
                            &case,
                            &model,
                            &judge_model,
                            (*sys_prompt).as_deref(),
                            threshold,
                            judge_temperature,
                            &base_url,
                        )
                        .await
                    };
                    (i, result)
                });
                handles.push(handle);
            }

            // Collect results in order
            let mut indexed_results: Vec<(usize, Result<CaseResult>)> = Vec::new();
            for handle in handles {
                indexed_results.push(handle.await?);
            }
            indexed_results.sort_by_key(|(i, _)| *i);

            let mut results = Vec::new();
            for (_, result) in indexed_results {
                let result = result?;
                if !json {
                    print_result(&result);
                }
                results.push(result);
            }

            // Compute summary
            let passed = results.iter().filter(|r| r.passed).count();
            let failed = results.len() - passed;
            let mean_score =
                results.iter().map(|r| r.score).sum::<f64>() / results.len().max(1) as f64;
            let total_latency: u64 = results.iter().map(|r| r.latency_ms).sum();
            let mut latencies: Vec<u64> = results.iter().map(|r| r.latency_ms).collect();
            latencies.sort();

            // Regression detection
            let regressions = if let Some(baseline_path) = &baseline {
                let baseline_content = std::fs::read_to_string(baseline_path)
                    .context("Failed to read baseline file")?;
                let baseline_results: RunResults = serde_json::from_str(&baseline_content)
                    .context("Failed to parse baseline JSON")?;

                let mut regs = Vec::new();
                for result in &results {
                    if baseline_results
                        .cases
                        .iter()
                        .find(|c| c.id == result.id)
                        .is_some_and(|base_case| base_case.passed && !result.passed)
                    {
                        regs.push(result.id.clone());
                    }
                }
                regs
            } else {
                Vec::new()
            };

            let summary = Summary {
                total: results.len(),
                passed,
                failed,
                pass_rate: passed as f64 / results.len().max(1) as f64,
                mean_score,
                total_latency_ms: total_latency,
                p50_latency_ms: percentile(&latencies, 50.0),
                p95_latency_ms: percentile(&latencies, 95.0),
                regressions: regressions.clone(),
            };

            let run_results = RunResults {
                timestamp: chrono_now(),
                model: model.to_string(),
                judge_model: judge_model.to_string(),
                threshold,
                cases: results,
                summary,
            };

            if json {
                println!("{}", serde_json::to_string_pretty(&run_results)?);
            } else {
                eprintln!();
                eprintln!(
                    "Results: {}/{} passed ({:.1}%)",
                    passed,
                    case_count,
                    run_results.summary.pass_rate * 100.0
                );
                eprintln!("  Mean score: {:.2}", run_results.summary.mean_score);
                eprintln!(
                    "  Latency: p50={:.1}s p95={:.1}s",
                    run_results.summary.p50_latency_ms as f64 / 1000.0,
                    run_results.summary.p95_latency_ms as f64 / 1000.0
                );
                if !regressions.is_empty() {
                    eprintln!(
                        "  \x1b[31mRegressions: {}\x1b[0m",
                        regressions.join(", ")
                    );
                }
                if failed > 0 {
                    let failed_ids: Vec<_> = run_results
                        .cases
                        .iter()
                        .filter(|c| !c.passed)
                        .map(|c| c.id.as_str())
                        .collect();
                    eprintln!("  Failures: {}", failed_ids.join(", "));
                }
            }

            // Exit code
            if failed > 0 {
                std::process::exit(1);
            }
        }
    }

    Ok(())
}

fn chrono_now() -> String {
    let output = std::process::Command::new("date")
        .args(["+%Y-%m-%dT%H:%M:%S%z"])
        .output()
        .ok();
    output
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}
