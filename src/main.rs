use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Instant;

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

        /// Model to use for generating responses (default: claude-sonnet-4-6)
        #[arg(long, default_value = "claude-sonnet-4-6")]
        model: String,

        /// Model to use for LLM-as-judge grading (default: claude-sonnet-4-6)
        #[arg(long, default_value = "claude-sonnet-4-6")]
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
    },
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
    diff: String,
    document_context: String,
    expected_behavior: Vec<String>,
    #[allow(dead_code)]
    #[serde(default = "default_grading")]
    grading: String,
    #[allow(dead_code)]
    #[serde(default)]
    notes: Option<String>,
}

fn default_grading() -> String {
    "llm-rubric".to_string()
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
    tokens_in: u64,
    tokens_out: u64,
    latency_ms: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct Summary {
    total: usize,
    passed: usize,
    failed: usize,
    pass_rate: f64,
    mean_score: f64,
    total_tokens: u64,
    total_latency_ms: u64,
    p50_latency_ms: u64,
    p95_latency_ms: u64,
    regressions: Vec<String>,
}

// --- Claude API ---

#[derive(Debug, Serialize)]
struct ApiRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<ApiMessage>,
}

#[derive(Debug, Serialize)]
struct ApiMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ApiResponse {
    content: Vec<ContentBlock>,
    usage: Usage,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    text: String,
}

#[derive(Debug, Deserialize)]
struct Usage {
    input_tokens: u64,
    output_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct JudgeOutput {
    score: f64,
    reasoning: String,
}

async fn call_claude(
    client: &reqwest::Client,
    api_key: &str,
    model: &str,
    system: Option<&str>,
    prompt: &str,
) -> Result<(String, u64, u64)> {
    let request = ApiRequest {
        model: model.to_string(),
        max_tokens: 4096,
        system: system.map(|s| s.to_string()),
        messages: vec![ApiMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }],
    };

    let resp = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&request)
        .send()
        .await
        .context("Failed to send request to Claude API")?;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Claude API error {status}: {body}");
    }

    let api_resp: ApiResponse = resp.json().await.context("Failed to parse Claude response")?;
    let text = api_resp
        .content
        .first()
        .map(|c| c.text.clone())
        .unwrap_or_default();

    Ok((text, api_resp.usage.input_tokens, api_resp.usage.output_tokens))
}

// --- Eval logic ---

async fn run_case(
    client: &reqwest::Client,
    api_key: &str,
    case: &EvalCase,
    model: &str,
    judge_model: &str,
    system_prompt: Option<&str>,
    threshold: f64,
) -> Result<CaseResult> {
    let start = Instant::now();

    // Step 1: Generate response from the model
    let prompt = format!(
        "You are an AI agent responding to user edits in a markdown document session.\n\n\
         **Diff (user's changes):**\n```\n{}\n```\n\n\
         **Document context:**\n{}\n\n\
         Respond to the user's changes naturally. Address what they changed.",
        case.diff, case.document_context
    );

    let (response, tokens_in, tokens_out) =
        call_claude(client, api_key, model, system_prompt, &prompt).await?;

    let latency_ms = start.elapsed().as_millis() as u64;

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
        case.id, case.description, expected, response
    );

    let (judge_resp, judge_in, judge_out) =
        call_claude(client, api_key, judge_model, None, &judge_prompt).await?;

    let judge_output: JudgeOutput = serde_json::from_str(&judge_resp)
        .unwrap_or_else(|_| {
            // Try to extract JSON from response if it has markdown fences
            let cleaned = judge_resp
                .trim_start_matches("```json")
                .trim_start_matches("```")
                .trim_end_matches("```")
                .trim();
            serde_json::from_str(cleaned).unwrap_or(JudgeOutput {
                score: 0.0,
                reasoning: format!("Failed to parse judge response: {judge_resp}"),
            })
        });

    Ok(CaseResult {
        id: case.id.clone(),
        description: case.description.clone(),
        passed: judge_output.score >= threshold,
        score: judge_output.score,
        reasoning: judge_output.reasoning,
        response_preview: response.chars().take(200).collect(),
        tokens_in: tokens_in + judge_in,
        tokens_out: tokens_out + judge_out,
        latency_ms,
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
    let tokens = result.tokens_in + result.tokens_out;
    let latency = format!("{:.1}s", result.latency_ms as f64 / 1000.0);

    eprintln!(
        "  [{status}] {:<30} score={score}  tokens={tokens:<6}  latency={latency}",
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
        } => {
            // Load API key
            let api_key = std::env::var("ANTHROPIC_API_KEY")
                .context("ANTHROPIC_API_KEY environment variable not set")?;

            // Load cases
            let content =
                std::fs::read_to_string(&cases).context("Failed to read cases file")?;
            let cases_file: CasesFile =
                serde_yaml::from_str(&content).context("Failed to parse cases YAML")?;

            // Load system prompt if provided
            let sys_prompt = system_prompt
                .map(|p| std::fs::read_to_string(p).context("Failed to read system prompt"))
                .transpose()?;

            let client = reqwest::Client::new();
            let case_count = cases_file.cases.len();

            if !json {
                eprintln!(
                    "\nRunning {case_count} cases (model={model}, judge={judge_model}, threshold={threshold})...\n"
                );
            }

            let mut results = Vec::new();
            for case in &cases_file.cases {
                let result = run_case(
                    &client,
                    &api_key,
                    case,
                    &model,
                    &judge_model,
                    sys_prompt.as_deref(),
                    threshold,
                )
                .await?;

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
            let total_tokens: u64 = results.iter().map(|r| r.tokens_in + r.tokens_out).sum();
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
                total_tokens,
                total_latency_ms: total_latency,
                p50_latency_ms: percentile(&latencies, 50.0),
                p95_latency_ms: percentile(&latencies, 95.0),
                regressions: regressions.clone(),
            };

            let run_results = RunResults {
                timestamp: chrono_now(),
                model: model.clone(),
                judge_model: judge_model.clone(),
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
                eprintln!("  Tokens: {} total", run_results.summary.total_tokens);
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
