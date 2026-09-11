use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::Serialize;

use crate::error::ProviderError;
use crate::provider::Provider;
use crate::types::{
    ContentBlock, Effort, FinishReason, ModelId, Request, Response, Role, StreamChunk,
    StreamResponse, ThinkingConfig, ThinkingDisplay, ToolDefinition, ToolUse, Usage,
};

const API_BASE: &str = "https://api.anthropic.com";
const API_VERSION: &str = "2023-06-01";

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    base_url: String,
    model: ModelId,
    extra_headers: Vec<(String, String)>,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            base_url: API_BASE.to_string(),
            model: ModelId::new(model),
            extra_headers: Vec::new(),
        }
    }

    pub fn with_base_url(
        api_key: impl Into<String>,
        model: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: ModelId::new(model),
            extra_headers: Vec::new(),
        }
    }

    /// Add a header to every request this provider sends, e.g.
    /// `anthropic-workspace-id` for API keys that are not scoped to a single
    /// workspace, or an `anthropic-beta` feature flag.
    pub fn with_extra_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_headers.push((name.into(), value.into()));
        self
    }

    fn request_builder(&self, url: &str) -> reqwest::RequestBuilder {
        let mut builder = self
            .client
            .post(url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json");
        for (name, value) in &self.extra_headers {
            builder = builder.header(name, value);
        }
        builder
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    async fn complete(&self, req: &Request) -> Result<Response, ProviderError> {
        let url = format!("{}/v1/messages", self.base_url);
        let body = MessagesRequest::from_request(req, &self.model, false);

        let start = std::time::Instant::now();
        let resp = self.request_builder(&url).json(&body).send().await?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(ProviderError::Api {
                status: status.as_u16(),
                message: text,
            });
        }

        let raw: serde_json::Value = resp.json().await?;
        let latency = start.elapsed();
        parse_messages_response(raw, &self.model, latency)
    }

    async fn stream(&self, req: &Request) -> Result<StreamResponse<'_>, ProviderError> {
        let url = format!("{}/v1/messages", self.base_url);
        let body = MessagesRequest::from_request(req, &self.model, true);

        let resp = self.request_builder(&url).json(&body).send().await?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(ProviderError::Api {
                status: status.as_u16(),
                message: text,
            });
        }

        // Anthropic uses SSE: lines prefixed with "event:" and "data:".
        // We accumulate partial lines from the byte stream, then parse
        // complete SSE frames.
        let byte_stream = resp.bytes_stream();

        let stream = futures::stream::unfold(
            SseState {
                inner: Box::pin(byte_stream),
                buf: String::new(),
                pending: Vec::new(),
                done: false,
                ctx: SseParseCtx::default(),
            },
            |mut state| async move {
                if state.done {
                    return None;
                }

                loop {
                    // Try to extract a complete SSE frame from the buffer.
                    let SseState { buf, ctx, .. } = &mut state;
                    if let Some(chunk) = try_parse_sse_frame(buf, ctx) {
                        match chunk {
                            SseFrame::Chunk(c) => return Some((c, state)),
                            SseFrame::Done(usage) => {
                                state.done = true;
                                return Some((StreamChunk::Done { usage }, state));
                            }
                            SseFrame::Skip => continue,
                        }
                    }

                    // Need more data from the network.
                    match state.inner.next().await {
                        Some(Ok(bytes)) => {
                            if let Err(e) =
                                push_chunk_utf8(&mut state.buf, &mut state.pending, &bytes)
                            {
                                state.done = true;
                                return Some((StreamChunk::Error(e.to_string()), state));
                            }
                        }
                        Some(Err(e)) => {
                            state.done = true;
                            return Some((StreamChunk::Error(e.to_string()), state));
                        }
                        None => {
                            state.done = true;
                            // Stream ended without a message_stop — still
                            // surface a Done so consumers don't hang.
                            return Some((StreamChunk::Done { usage: None }, state));
                        }
                    }
                }
            },
        );

        Ok(Box::pin(stream))
    }

    fn model_id(&self) -> &ModelId {
        &self.model
    }
}

// ---------------------------------------------------------------------------
// SSE parsing state
// ---------------------------------------------------------------------------

struct SseState {
    inner:
        std::pin::Pin<Box<dyn futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send>>,
    buf: String,
    /// Trailing bytes of a UTF-8 character split across network chunks.
    pending: Vec<u8>,
    done: bool,
    ctx: SseParseCtx,
}

/// Cross-frame parse state: what earlier SSE frames established about the
/// message in flight. Frame parsing itself stays line-local; these two facts
/// are the only ones that outlive a frame.
#[derive(Default)]
struct SseParseCtx {
    /// From `message_start` — the closing `message_delta` usage carries only
    /// `output_tokens`, so the input count must be remembered until then or
    /// every streamed response reports zero input tokens.
    input_tokens: Option<u32>,
    /// True while inside a thinking-family content block (`thinking`,
    /// `summarized_thinking`, …). Reasoning that streams as plain
    /// `text_delta`s inside such a block must surface as `Thinking`, never
    /// `Delta` — otherwise hosts persist the reasoning into the answer.
    in_thinking_block: bool,
}

/// Append a network chunk to the text buffer. HTTP chunk boundaries fall on
/// arbitrary byte offsets, so a multi-byte UTF-8 character can straddle two
/// chunks; the incomplete tail is carried in `pending` until the rest arrives.
fn push_chunk_utf8(
    buf: &mut String,
    pending: &mut Vec<u8>,
    chunk: &[u8],
) -> Result<(), std::str::Utf8Error> {
    pending.extend_from_slice(chunk);
    match std::str::from_utf8(pending) {
        Ok(s) => {
            buf.push_str(s);
            pending.clear();
            Ok(())
        }
        // `error_len() == None` means the buffer ends mid-character: decode
        // the valid prefix and keep the tail for the next chunk.
        Err(e) if e.error_len().is_none() => {
            let valid = e.valid_up_to();
            buf.push_str(std::str::from_utf8(&pending[..valid]).expect("prefix is valid UTF-8"));
            pending.drain(..valid);
            Ok(())
        }
        Err(e) => Err(e),
    }
}

#[derive(Debug)]
enum SseFrame {
    Chunk(StreamChunk),
    Done(Option<Usage>),
    Skip,
}

/// Try to consume one complete SSE frame (`event: ...\ndata: ...\n\n`) from
/// the buffer. Returns `None` if there isn't a complete frame yet. `ctx`
/// carries the little cross-frame state a frame can establish (input token
/// count, whether we're inside a thinking block).
fn try_parse_sse_frame(buf: &mut String, ctx: &mut SseParseCtx) -> Option<SseFrame> {
    // SSE frames are terminated by a blank line (\n\n).
    let frame_end = buf.find("\n\n")?;
    let frame: String = buf.drain(..frame_end + 2).collect();

    let mut event_type = "";
    let mut data = String::new();

    for line in frame.lines() {
        if let Some(val) = line.strip_prefix("event: ") {
            event_type = val.trim();
        } else if let Some(val) = line.strip_prefix("event:") {
            event_type = val.trim();
        } else if let Some(val) = line.strip_prefix("data: ") {
            data.push_str(val);
        } else if let Some(val) = line.strip_prefix("data:") {
            data.push_str(val);
        }
    }

    match event_type {
        "content_block_delta" => {
            let v: serde_json::Value = serde_json::from_str(&data).ok()?;
            // Extended thinking streams a `thinking_delta` carrying `delta.thinking`
            // (and a trailing `signature_delta` we ignore). Surface reasoning text
            // as a distinct chunk so hosts can render it apart from the answer.
            if let Some(thinking) = v["delta"]["thinking"].as_str() {
                return if thinking.is_empty() {
                    Some(SseFrame::Skip)
                } else {
                    Some(SseFrame::Chunk(StreamChunk::Thinking(thinking.to_string())))
                };
            }
            let text = v["delta"]["text"].as_str().unwrap_or("").to_string();
            if text.is_empty() {
                Some(SseFrame::Skip)
            } else if ctx.in_thinking_block {
                // A thinking-family block whose deltas arrive as plain
                // `text_delta`s (display-shape dependent). It is reasoning,
                // not answer — a host that buffered it as Delta would persist
                // the thinking into the reply.
                Some(SseFrame::Chunk(StreamChunk::Thinking(text)))
            } else {
                Some(SseFrame::Chunk(StreamChunk::Delta(text)))
            }
        }
        "message_delta" => {
            // Contains stop_reason and final usage — output side only. The
            // input count arrived in `message_start` and was parked in `ctx`.
            let v: serde_json::Value = serde_json::from_str(&data).ok()?;
            let output_tokens = v["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32;
            Some(SseFrame::Done(Some(Usage {
                input_tokens: ctx.input_tokens.unwrap_or(0),
                output_tokens,
            })))
        }
        "message_stop" => Some(SseFrame::Skip),
        "message_start" => {
            let v: serde_json::Value = serde_json::from_str(&data).ok()?;
            if let Some(n) = v["message"]["usage"]["input_tokens"].as_u64() {
                ctx.input_tokens = Some(n as u32);
            }
            Some(SseFrame::Skip)
        }
        "content_block_start" => {
            let v: serde_json::Value = serde_json::from_str(&data).ok()?;
            let block_type = v["content_block"]["type"].as_str().unwrap_or("");
            ctx.in_thinking_block = block_type.contains("thinking");
            Some(SseFrame::Skip)
        }
        "content_block_stop" => {
            ctx.in_thinking_block = false;
            Some(SseFrame::Skip)
        }
        "ping" => Some(SseFrame::Skip),
        "error" => {
            let v: serde_json::Value = serde_json::from_str(&data).ok()?;
            let msg = v["error"]["message"]
                .as_str()
                .unwrap_or("unknown error")
                .to_string();
            Some(SseFrame::Chunk(StreamChunk::Error(msg)))
        }
        _ => Some(SseFrame::Skip),
    }
}

// ---------------------------------------------------------------------------
// Wire types — Anthropic Messages API
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct MessagesRequest {
    model: String,
    messages: Vec<ApiMessage>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop_sequences: Vec<String>,
    // Anthropic's tool schema matches our ToolDefinition field-for-field
    // ({name, description, input_schema}), so we serialize it directly.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<serde_json::Value>,
    stream: bool,
}

#[derive(Serialize)]
struct ApiMessage {
    role: String,
    content: serde_json::Value,
}

/// Serialize a message's content blocks into Anthropic's content-array format.
fn blocks_to_anthropic(content: &[ContentBlock]) -> serde_json::Value {
    let arr: Vec<serde_json::Value> = content
        .iter()
        .map(|b| match b {
            ContentBlock::Text { text } => serde_json::json!({ "type": "text", "text": text }),
            ContentBlock::ToolUse { id, name, input } => serde_json::json!({
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": input,
            }),
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                let mut v = serde_json::json!({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content,
                });
                if *is_error {
                    v["is_error"] = serde_json::Value::Bool(true);
                }
                v
            }
        })
        .collect();
    serde_json::Value::Array(arr)
}

fn display_str(d: ThinkingDisplay) -> &'static str {
    match d {
        ThinkingDisplay::Summarized => "summarized",
        ThinkingDisplay::Omitted => "omitted",
    }
}

fn effort_str(e: Effort) -> &'static str {
    match e {
        Effort::Low => "low",
        Effort::Medium => "medium",
        Effort::High => "high",
        Effort::XHigh => "xhigh",
        Effort::Max => "max",
    }
}

impl MessagesRequest {
    fn from_request(req: &Request, default_model: &ModelId, stream: bool) -> Self {
        let model = if req.model.as_str() == "default" {
            default_model.as_str().to_string()
        } else {
            req.model.as_str().to_string()
        };

        // Anthropic requires max_tokens. Default to 4096 if unset.
        let max_tokens = req.max_tokens.unwrap_or(4096);

        // Collect system messages into the top-level `system` field.
        // Anthropic doesn't allow role:"system" in the messages array.
        let mut system_parts: Vec<String> = Vec::new();
        if let Some(s) = &req.system {
            system_parts.push(s.clone());
        }

        let mut messages: Vec<ApiMessage> = Vec::new();
        for m in &req.messages {
            match m.role {
                // System messages contribute only their text to the top-level
                // system field (Anthropic disallows role:"system" in the array).
                Role::System => system_parts.push(m.text()),
                Role::User => messages.push(ApiMessage {
                    role: "user".into(),
                    content: blocks_to_anthropic(&m.content),
                }),
                Role::Assistant => messages.push(ApiMessage {
                    role: "assistant".into(),
                    content: blocks_to_anthropic(&m.content),
                }),
            }
        }

        let system = if system_parts.is_empty() {
            None
        } else {
            Some(system_parts.join("\n"))
        };

        let thinking = req.thinking.as_ref().map(|t| match t {
            ThinkingConfig::Adaptive { display, .. } => serde_json::json!({
                "type": "adaptive",
                "display": display_str(*display),
            }),
            ThinkingConfig::Disabled => serde_json::json!({ "type": "disabled" }),
            ThinkingConfig::Enabled { budget_tokens } => serde_json::json!({
                "type": "enabled",
                "budget_tokens": budget_tokens,
            }),
        });

        // `effort` rides on the adaptive variant but serializes to the separate
        // top-level `output_config.effort` field.
        let output_config = match &req.thinking {
            Some(ThinkingConfig::Adaptive {
                effort: Some(e), ..
            }) => Some(serde_json::json!({ "effort": effort_str(*e) })),
            _ => None,
        };

        // Anthropic rejects a custom `temperature` when extended thinking is
        // active (adaptive or legacy budgeted). Drop it in that case.
        let temperature = match &req.thinking {
            Some(ThinkingConfig::Adaptive { .. }) | Some(ThinkingConfig::Enabled { .. }) => None,
            _ => req.temperature,
        };

        Self {
            model,
            messages,
            max_tokens,
            system,
            temperature,
            stop_sequences: req.stop.clone(),
            tools: req.tools.clone(),
            thinking,
            output_config,
            stream,
        }
    }
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

fn parse_messages_response(
    raw: serde_json::Value,
    default_model: &ModelId,
    latency: std::time::Duration,
) -> Result<Response, ProviderError> {
    // Walk the content blocks: concatenate text, collect tool_use calls.
    let mut content = String::new();
    let mut tool_calls: Vec<ToolUse> = Vec::new();
    if let Some(blocks) = raw["content"].as_array() {
        for b in blocks {
            match b["type"].as_str() {
                Some("text") => content.push_str(b["text"].as_str().unwrap_or("")),
                Some("tool_use") => tool_calls.push(ToolUse {
                    id: b["id"].as_str().unwrap_or("").to_string(),
                    name: b["name"].as_str().unwrap_or("").to_string(),
                    input: b["input"].clone(),
                }),
                _ => {}
            }
        }
    }

    let stop_reason = raw["stop_reason"].as_str().unwrap_or("end_turn");
    let finish_reason = match stop_reason {
        "end_turn" | "stop_sequence" => FinishReason::Stop,
        "max_tokens" => FinishReason::MaxTokens,
        "tool_use" => FinishReason::ToolUse,
        other => FinishReason::Other(other.into()),
    };

    let model_str = raw["model"].as_str().unwrap_or(default_model.as_str());

    let usage = Usage {
        input_tokens: raw["usage"]["input_tokens"].as_u64().unwrap_or(0) as u32,
        output_tokens: raw["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32,
    };

    Ok(Response {
        content,
        tool_calls,
        usage,
        model: ModelId::new(model_str),
        finish_reason,
        latency,
        raw,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;
    use futures::StreamExt;
    use std::time::Duration;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    // ── push_chunk_utf8 ──────────────────────────────────────────────────

    #[test]
    fn utf8_char_split_across_chunks() {
        // "…" is E2 80 A6; a chunk boundary in the middle must not error.
        let text = "context truncated…done".as_bytes();
        let (a, b) = text.split_at(text.iter().position(|&b| b == 0xE2).unwrap() + 1);
        let mut buf = String::new();
        let mut pending = Vec::new();
        push_chunk_utf8(&mut buf, &mut pending, a).unwrap();
        assert_eq!(buf, "context truncated");
        assert_eq!(pending, [0xE2]);
        push_chunk_utf8(&mut buf, &mut pending, b).unwrap();
        assert_eq!(buf, "context truncated…done");
        assert!(pending.is_empty());
    }

    #[test]
    fn utf8_char_split_one_byte_per_chunk() {
        let mut buf = String::new();
        let mut pending = Vec::new();
        for byte in "🎉".as_bytes() {
            push_chunk_utf8(&mut buf, &mut pending, &[*byte]).unwrap();
        }
        assert_eq!(buf, "🎉");
        assert!(pending.is_empty());
    }

    #[test]
    fn utf8_genuinely_invalid_bytes_error() {
        let mut buf = String::new();
        let mut pending = Vec::new();
        // 0xE2 followed by an ASCII byte can never form a valid character.
        assert!(push_chunk_utf8(&mut buf, &mut pending, &[0xE2, b'x']).is_err());
    }

    // ── parse_messages_response ──────────────────────────────────────────

    #[test]
    fn parse_response_full() {
        let raw = serde_json::json!({
            "content": [
                { "type": "text", "text": "Hello!" }
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 15, "output_tokens": 8 }
        });
        let resp = parse_messages_response(raw, &ModelId::new("fallback"), Duration::from_secs(2))
            .unwrap();
        assert_eq!(resp.content, "Hello!");
        assert_eq!(resp.model.as_str(), "claude-sonnet-4-20250514");
        assert_eq!(resp.finish_reason, FinishReason::Stop);
        assert_eq!(resp.usage.input_tokens, 15);
        assert_eq!(resp.usage.output_tokens, 8);
        assert_eq!(resp.latency, Duration::from_secs(2));
    }

    #[test]
    fn parse_response_stop_sequence() {
        let raw = serde_json::json!({ "stop_reason": "stop_sequence", "content": [] });
        let resp = parse_messages_response(raw, &ModelId::new("f"), Duration::ZERO).unwrap();
        assert_eq!(resp.finish_reason, FinishReason::Stop);
    }

    #[test]
    fn parse_response_max_tokens() {
        let raw = serde_json::json!({ "stop_reason": "max_tokens", "content": [] });
        let resp = parse_messages_response(raw, &ModelId::new("f"), Duration::ZERO).unwrap();
        assert_eq!(resp.finish_reason, FinishReason::MaxTokens);
    }

    #[test]
    fn parse_response_other_stop() {
        let raw = serde_json::json!({ "stop_reason": "custom", "content": [] });
        let resp = parse_messages_response(raw, &ModelId::new("f"), Duration::ZERO).unwrap();
        assert_eq!(resp.finish_reason, FinishReason::Other("custom".into()));
    }

    #[test]
    fn parse_response_missing_stop_reason() {
        let raw = serde_json::json!({ "content": [] });
        let resp = parse_messages_response(raw, &ModelId::new("f"), Duration::ZERO).unwrap();
        assert_eq!(resp.finish_reason, FinishReason::Stop); // defaults to end_turn
    }

    #[test]
    fn parse_response_no_content() {
        let raw = serde_json::json!({});
        let resp = parse_messages_response(raw, &ModelId::new("f"), Duration::ZERO).unwrap();
        assert_eq!(resp.content, "");
    }

    #[test]
    fn parse_response_mixed_blocks() {
        let raw = serde_json::json!({
            "content": [
                { "type": "text", "text": "A" },
                { "type": "tool_use", "id": "x" },
                { "type": "text", "text": "B" },
            ]
        });
        let resp = parse_messages_response(raw, &ModelId::new("f"), Duration::ZERO).unwrap();
        assert_eq!(resp.content, "AB");
    }

    #[test]
    fn parse_response_missing_model() {
        let raw = serde_json::json!({ "content": [] });
        let resp = parse_messages_response(raw, &ModelId::new("fallback"), Duration::ZERO).unwrap();
        assert_eq!(resp.model.as_str(), "fallback");
    }

    #[test]
    fn parse_response_missing_usage() {
        let raw = serde_json::json!({ "content": [] });
        let resp = parse_messages_response(raw, &ModelId::new("f"), Duration::ZERO).unwrap();
        assert_eq!(resp.usage.input_tokens, 0);
        assert_eq!(resp.usage.output_tokens, 0);
    }

    // ── try_parse_sse_frame ──────────────────────────────────────────────

    #[test]
    fn sse_incomplete_frame() {
        let mut ctx = SseParseCtx::default();
        let mut buf = "event: ping\ndata: {}\n".to_string(); // no \n\n
        let original_len = buf.len();
        assert!(try_parse_sse_frame(&mut buf, &mut ctx).is_none());
        assert_eq!(buf.len(), original_len); // buffer not drained
    }

    #[test]
    fn sse_content_block_delta() {
        let mut ctx = SseParseCtx::default();
        let mut buf =
            "event: content_block_delta\ndata: {\"delta\":{\"text\":\"hi\"}}\n\n".to_string();
        match try_parse_sse_frame(&mut buf, &mut ctx) {
            Some(SseFrame::Chunk(StreamChunk::Delta(t))) => assert_eq!(t, "hi"),
            other => panic!("expected Chunk(Delta), got {other:?}"),
        }
        assert!(buf.is_empty());
    }

    #[test]
    fn sse_content_block_delta_thinking() {
        let mut ctx = SseParseCtx::default();
        let mut buf =
            "event: content_block_delta\ndata: {\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"let me consider\"}}\n\n"
                .to_string();
        match try_parse_sse_frame(&mut buf, &mut ctx) {
            Some(SseFrame::Chunk(StreamChunk::Thinking(t))) => assert_eq!(t, "let me consider"),
            other => panic!("expected Chunk(Thinking), got {other:?}"),
        }
        assert!(buf.is_empty());
    }

    #[test]
    fn sse_content_block_delta_signature_is_skipped() {
        // The trailing signature_delta of a thinking block has neither `thinking`
        // nor `text`, so it must be skipped rather than emitted.
        let mut ctx = SseParseCtx::default();
        let mut buf =
            "event: content_block_delta\ndata: {\"delta\":{\"type\":\"signature_delta\",\"signature\":\"abc\"}}\n\n"
                .to_string();
        match try_parse_sse_frame(&mut buf, &mut ctx) {
            Some(SseFrame::Skip) => {}
            other => panic!("expected Skip, got {other:?}"),
        }
        assert!(buf.is_empty());
    }

    #[test]
    fn sse_content_block_delta_empty_text() {
        let mut ctx = SseParseCtx::default();
        let mut buf =
            "event: content_block_delta\ndata: {\"delta\":{\"text\":\"\"}}\n\n".to_string();
        match try_parse_sse_frame(&mut buf, &mut ctx) {
            Some(SseFrame::Skip) => {}
            other => panic!("expected Skip, got {other:?}"),
        }
    }

    #[test]
    fn sse_message_delta() {
        let mut ctx = SseParseCtx::default();
        let mut buf =
            "event: message_delta\ndata: {\"usage\":{\"output_tokens\":42}}\n\n".to_string();
        match try_parse_sse_frame(&mut buf, &mut ctx) {
            Some(SseFrame::Done(Some(usage))) => {
                // No message_start seen (degenerate stream): input honestly 0.
                assert_eq!(usage.input_tokens, 0);
                assert_eq!(usage.output_tokens, 42);
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[test]
    fn sse_input_tokens_carry_from_message_start_to_done() {
        // Anthropic reports input tokens ONLY in message_start; the closing
        // message_delta carries just output usage. Dropping the start's count
        // is how every streamed chat came out `input_tokens: 0`.
        let mut ctx = SseParseCtx::default();
        let mut buf = concat!(
            "event: message_start\n",
            "data: {\"message\":{\"usage\":{\"input_tokens\":4321,\"output_tokens\":1}}}\n\n",
            "event: message_delta\n",
            "data: {\"usage\":{\"output_tokens\":42}}\n\n",
        )
        .to_string();
        assert!(matches!(
            try_parse_sse_frame(&mut buf, &mut ctx),
            Some(SseFrame::Skip)
        ));
        match try_parse_sse_frame(&mut buf, &mut ctx) {
            Some(SseFrame::Done(Some(usage))) => {
                assert_eq!(
                    usage.input_tokens, 4321,
                    "message_start count must be carried"
                );
                assert_eq!(usage.output_tokens, 42);
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[test]
    fn sse_text_deltas_inside_a_thinking_block_are_thinking() {
        // Some display shapes stream reasoning as plain text_deltas inside a
        // `*thinking*`-typed content block. Those are reasoning, not answer —
        // a host that buffers Delta chunks would persist the thinking into
        // the reply. The block type governs classification.
        let mut ctx = SseParseCtx::default();
        let mut buf = concat!(
            "event: content_block_start\n",
            "data: {\"content_block\":{\"type\":\"summarized_thinking\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"delta\":{\"type\":\"text_delta\",\"text\":\"weighing options\"}}\n\n",
            "event: content_block_stop\n",
            "data: {}\n\n",
            "event: content_block_start\n",
            "data: {\"content_block\":{\"type\":\"text\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"delta\":{\"type\":\"text_delta\",\"text\":\"the answer\"}}\n\n",
        )
        .to_string();
        assert!(matches!(
            try_parse_sse_frame(&mut buf, &mut ctx),
            Some(SseFrame::Skip)
        ));
        match try_parse_sse_frame(&mut buf, &mut ctx) {
            Some(SseFrame::Chunk(StreamChunk::Thinking(t))) => assert_eq!(t, "weighing options"),
            other => panic!("expected Thinking inside the block, got {other:?}"),
        }
        assert!(matches!(
            try_parse_sse_frame(&mut buf, &mut ctx),
            Some(SseFrame::Skip)
        ));
        assert!(matches!(
            try_parse_sse_frame(&mut buf, &mut ctx),
            Some(SseFrame::Skip)
        ));
        match try_parse_sse_frame(&mut buf, &mut ctx) {
            Some(SseFrame::Chunk(StreamChunk::Delta(t))) => assert_eq!(t, "the answer"),
            other => panic!("expected Delta after the block closed, got {other:?}"),
        }
    }

    #[test]
    fn sse_message_stop() {
        let mut ctx = SseParseCtx::default();
        let mut buf = "event: message_stop\ndata: {}\n\n".to_string();
        match try_parse_sse_frame(&mut buf, &mut ctx) {
            Some(SseFrame::Skip) => {}
            other => panic!("expected Skip, got {other:?}"),
        }
    }

    #[test]
    fn sse_message_start() {
        let mut ctx = SseParseCtx::default();
        let mut buf = "event: message_start\ndata: {}\n\n".to_string();
        match try_parse_sse_frame(&mut buf, &mut ctx) {
            Some(SseFrame::Skip) => {}
            other => panic!("expected Skip, got {other:?}"),
        }
    }

    #[test]
    fn sse_content_block_start() {
        let mut ctx = SseParseCtx::default();
        let mut buf = "event: content_block_start\ndata: {}\n\n".to_string();
        match try_parse_sse_frame(&mut buf, &mut ctx) {
            Some(SseFrame::Skip) => {}
            other => panic!("expected Skip, got {other:?}"),
        }
    }

    #[test]
    fn sse_content_block_stop() {
        let mut ctx = SseParseCtx::default();
        let mut buf = "event: content_block_stop\ndata: {}\n\n".to_string();
        match try_parse_sse_frame(&mut buf, &mut ctx) {
            Some(SseFrame::Skip) => {}
            other => panic!("expected Skip, got {other:?}"),
        }
    }

    #[test]
    fn sse_ping() {
        let mut ctx = SseParseCtx::default();
        let mut buf = "event: ping\ndata: {}\n\n".to_string();
        match try_parse_sse_frame(&mut buf, &mut ctx) {
            Some(SseFrame::Skip) => {}
            other => panic!("expected Skip, got {other:?}"),
        }
    }

    #[test]
    fn sse_error_event() {
        let mut ctx = SseParseCtx::default();
        let mut buf =
            "event: error\ndata: {\"error\":{\"message\":\"overloaded\"}}\n\n".to_string();
        match try_parse_sse_frame(&mut buf, &mut ctx) {
            Some(SseFrame::Chunk(StreamChunk::Error(msg))) => assert_eq!(msg, "overloaded"),
            other => panic!("expected Error chunk, got {other:?}"),
        }
    }

    #[test]
    fn sse_error_no_message() {
        let mut ctx = SseParseCtx::default();
        let mut buf = "event: error\ndata: {\"error\":{}}\n\n".to_string();
        match try_parse_sse_frame(&mut buf, &mut ctx) {
            Some(SseFrame::Chunk(StreamChunk::Error(msg))) => assert_eq!(msg, "unknown error"),
            other => panic!("expected Error chunk with unknown, got {other:?}"),
        }
    }

    #[test]
    fn sse_unknown_event() {
        let mut ctx = SseParseCtx::default();
        let mut buf = "event: custom_thing\ndata: {}\n\n".to_string();
        match try_parse_sse_frame(&mut buf, &mut ctx) {
            Some(SseFrame::Skip) => {}
            other => panic!("expected Skip, got {other:?}"),
        }
    }

    #[test]
    fn sse_no_space_after_colon() {
        let mut ctx = SseParseCtx::default();
        let mut buf =
            "event:content_block_delta\ndata:{\"delta\":{\"text\":\"x\"}}\n\n".to_string();
        match try_parse_sse_frame(&mut buf, &mut ctx) {
            Some(SseFrame::Chunk(StreamChunk::Delta(t))) => assert_eq!(t, "x"),
            other => panic!("expected Delta, got {other:?}"),
        }
    }

    #[test]
    fn sse_invalid_json_returns_none() {
        let mut ctx = SseParseCtx::default();
        let mut buf = "event: content_block_delta\ndata: not-json\n\n".to_string();
        // serde_json::from_str fails, .ok()? returns None
        assert!(try_parse_sse_frame(&mut buf, &mut ctx).is_none());
    }

    #[test]
    fn sse_drains_buffer() {
        let mut ctx = SseParseCtx::default();
        let mut buf = "event: ping\ndata: {}\n\nevent: message_stop\ndata: {}\n\n".to_string();
        try_parse_sse_frame(&mut buf, &mut ctx); // consume first frame
        assert!(buf.starts_with("event: message_stop"));
        try_parse_sse_frame(&mut buf, &mut ctx); // consume second frame
        assert!(buf.is_empty());
    }

    // ── MessagesRequest::from_request ────────────────────────────────────

    #[test]
    fn msg_request_default_model() {
        let req = Request::default();
        let mr = MessagesRequest::from_request(&req, &ModelId::new("claude-3"), false);
        assert_eq!(mr.model, "claude-3");
    }

    #[test]
    fn msg_request_explicit_model() {
        let req = Request {
            model: ModelId::new("claude-opus"),
            ..Default::default()
        };
        let mr = MessagesRequest::from_request(&req, &ModelId::new("claude-3"), false);
        assert_eq!(mr.model, "claude-opus");
    }

    #[test]
    fn msg_request_max_tokens_default() {
        let req = Request::default();
        let mr = MessagesRequest::from_request(&req, &ModelId::new("m"), false);
        assert_eq!(mr.max_tokens, 4096);
    }

    #[test]
    fn msg_request_max_tokens_explicit() {
        let req = Request {
            max_tokens: Some(1000),
            ..Default::default()
        };
        let mr = MessagesRequest::from_request(&req, &ModelId::new("m"), false);
        assert_eq!(mr.max_tokens, 1000);
    }

    #[test]
    fn msg_request_thinking_enabled_serializes_and_drops_temperature() {
        let req = Request {
            max_tokens: Some(8192),
            temperature: Some(0.7), // must be dropped when thinking is enabled
            thinking: Some(ThinkingConfig::Enabled {
                budget_tokens: 4096,
            }),
            ..Default::default()
        };
        let mr = MessagesRequest::from_request(&req, &ModelId::new("m"), false);
        assert!(mr.temperature.is_none());
        let v = serde_json::to_value(&mr).unwrap();
        assert_eq!(v["thinking"]["type"], "enabled");
        assert_eq!(v["thinking"]["budget_tokens"], 4096);
        assert!(v.get("temperature").is_none());
    }

    #[test]
    fn msg_request_thinking_adaptive_serializes_thinking_and_effort() {
        let req = Request {
            max_tokens: Some(8192),
            temperature: Some(0.7), // dropped when thinking is active
            thinking: Some(ThinkingConfig::Adaptive {
                display: ThinkingDisplay::Summarized,
                effort: Some(Effort::Medium),
            }),
            ..Default::default()
        };
        let mr = MessagesRequest::from_request(&req, &ModelId::new("m"), false);
        assert!(mr.temperature.is_none());
        let v = serde_json::to_value(&mr).unwrap();
        assert_eq!(v["thinking"]["type"], "adaptive");
        assert_eq!(v["thinking"]["display"], "summarized");
        assert_eq!(v["output_config"]["effort"], "medium");
        assert!(v.get("temperature").is_none());
    }

    #[test]
    fn msg_request_thinking_adaptive_without_effort_omits_output_config() {
        let req = Request {
            thinking: Some(ThinkingConfig::Adaptive {
                display: ThinkingDisplay::Omitted,
                effort: None,
            }),
            ..Default::default()
        };
        let mr = MessagesRequest::from_request(&req, &ModelId::new("m"), false);
        let v = serde_json::to_value(&mr).unwrap();
        assert_eq!(v["thinking"]["display"], "omitted");
        assert!(v.get("output_config").is_none());
    }

    #[test]
    fn msg_request_thinking_disabled_serializes() {
        let req = Request {
            thinking: Some(ThinkingConfig::Disabled),
            ..Default::default()
        };
        let mr = MessagesRequest::from_request(&req, &ModelId::new("m"), false);
        let v = serde_json::to_value(&mr).unwrap();
        assert_eq!(v["thinking"]["type"], "disabled");
    }

    #[test]
    fn msg_request_no_thinking_omits_field() {
        let req = Request::default();
        let mr = MessagesRequest::from_request(&req, &ModelId::new("m"), false);
        let v = serde_json::to_value(&mr).unwrap();
        assert!(v.get("thinking").is_none());
    }

    #[test]
    fn msg_request_system_combined() {
        let req = Request {
            system: Some("A".into()),
            messages: vec![Message::system("B"), Message::user("hi")],
            ..Default::default()
        };
        let mr = MessagesRequest::from_request(&req, &ModelId::new("m"), false);
        assert_eq!(mr.system, Some("A\nB".into()));
        // System message should NOT appear in messages array
        assert_eq!(mr.messages.len(), 1);
        assert_eq!(mr.messages[0].role, "user");
    }

    #[test]
    fn msg_request_no_system() {
        let req = Request {
            messages: vec![Message::user("hi")],
            ..Default::default()
        };
        let mr = MessagesRequest::from_request(&req, &ModelId::new("m"), false);
        assert!(mr.system.is_none());
    }

    #[test]
    fn msg_request_system_messages_filtered() {
        let req = Request {
            messages: vec![
                Message::system("sys"),
                Message::user("usr"),
                Message::assistant("ast"),
            ],
            ..Default::default()
        };
        let mr = MessagesRequest::from_request(&req, &ModelId::new("m"), false);
        assert_eq!(mr.messages.len(), 2);
        assert_eq!(mr.messages[0].role, "user");
        assert_eq!(mr.messages[1].role, "assistant");
        assert_eq!(mr.system, Some("sys".into()));
    }

    #[test]
    fn msg_request_stream_flag() {
        let req = Request::default();
        assert!(MessagesRequest::from_request(&req, &ModelId::new("m"), true).stream);
        assert!(!MessagesRequest::from_request(&req, &ModelId::new("m"), false).stream);
    }

    // ── Tools ────────────────────────────────────────────────────────────

    #[test]
    fn msg_request_serializes_tools() {
        let req = Request {
            tools: vec![crate::types::ToolDefinition::new(
                "get_weather",
                "Get weather",
                serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}),
            )],
            messages: vec![Message::user("hi")],
            ..Default::default()
        };
        let mr = MessagesRequest::from_request(&req, &ModelId::new("m"), false);
        let body = serde_json::to_value(&mr).unwrap();
        assert_eq!(body["tools"][0]["name"], "get_weather");
        assert_eq!(body["tools"][0]["description"], "Get weather");
        assert_eq!(body["tools"][0]["input_schema"]["type"], "object");
        // User message content is serialized as a block array.
        assert_eq!(body["messages"][0]["content"][0]["type"], "text");
        assert_eq!(body["messages"][0]["content"][0]["text"], "hi");
    }

    #[test]
    fn msg_request_no_tools_field_when_empty() {
        let req = Request::default();
        let mr = MessagesRequest::from_request(&req, &ModelId::new("m"), false);
        let body = serde_json::to_value(&mr).unwrap();
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn msg_request_serializes_tool_result_block() {
        let req = Request {
            messages: vec![Message::tool_result("tu_1", "42", false)],
            ..Default::default()
        };
        let mr = MessagesRequest::from_request(&req, &ModelId::new("m"), false);
        let body = serde_json::to_value(&mr).unwrap();
        let block = &body["messages"][0]["content"][0];
        assert_eq!(block["type"], "tool_result");
        assert_eq!(block["tool_use_id"], "tu_1");
        assert_eq!(block["content"], "42");
        assert!(block.get("is_error").is_none());
    }

    #[test]
    fn parse_response_tool_use() {
        let raw = serde_json::json!({
            "content": [
                { "type": "text", "text": "Let me check." },
                { "type": "tool_use", "id": "tu_9", "name": "get_weather", "input": {"city": "SF"} },
            ],
            "stop_reason": "tool_use",
            "usage": { "input_tokens": 5, "output_tokens": 3 }
        });
        let resp = parse_messages_response(raw, &ModelId::new("f"), Duration::ZERO).unwrap();
        assert_eq!(resp.content, "Let me check.");
        assert_eq!(resp.finish_reason, FinishReason::ToolUse);
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].id, "tu_9");
        assert_eq!(resp.tool_calls[0].name, "get_weather");
        assert_eq!(resp.tool_calls[0].input["city"], "SF");
    }

    #[test]
    fn parse_response_no_tool_calls_when_text_only() {
        let raw = serde_json::json!({
            "content": [{ "type": "text", "text": "hi" }],
            "stop_reason": "end_turn"
        });
        let resp = parse_messages_response(raw, &ModelId::new("f"), Duration::ZERO).unwrap();
        assert!(resp.tool_calls.is_empty());
    }

    // ── Constructor tests ────────────────────────────────────────────────

    #[test]
    fn new_default_base_url() {
        let p = AnthropicProvider::new("key", "model");
        assert_eq!(p.base_url, "https://api.anthropic.com");
    }

    #[test]
    fn with_base_url_trims_slash() {
        let p = AnthropicProvider::with_base_url("k", "m", "http://host/");
        assert_eq!(p.base_url, "http://host");
    }

    #[test]
    fn model_id_returns_configured() {
        let p = AnthropicProvider::new("key", "claude-3");
        assert_eq!(p.model_id().as_str(), "claude-3");
    }

    // ── HTTP integration tests (wiremock) ────────────────────────────────

    fn anthropic_response_json() -> serde_json::Value {
        serde_json::json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{ "type": "text", "text": "Hello!" }],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 12, "output_tokens": 6 }
        })
    }

    #[tokio::test]
    async fn complete_success() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "test-key"))
            .and(header("anthropic-version", "2023-06-01"))
            .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_response_json()))
            .mount(&server)
            .await;

        let provider =
            AnthropicProvider::with_base_url("test-key", "claude-sonnet-4-20250514", server.uri());
        let resp = provider
            .complete(&Request {
                messages: vec![Message::user("hi")],
                ..Default::default()
            })
            .await
            .unwrap();

        assert_eq!(resp.content, "Hello!");
        assert_eq!(resp.usage.input_tokens, 12);
        assert_eq!(resp.usage.output_tokens, 6);
        assert_eq!(resp.model.as_str(), "claude-sonnet-4-20250514");
        assert!(resp.latency > Duration::ZERO);
    }

    #[tokio::test]
    async fn complete_sends_extra_headers() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "test-key"))
            .and(header("anthropic-workspace-id", "wrkspc_123"))
            .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_response_json()))
            .mount(&server)
            .await;

        let provider =
            AnthropicProvider::with_base_url("test-key", "claude-sonnet-4-20250514", server.uri())
                .with_extra_header("anthropic-workspace-id", "wrkspc_123");
        let resp = provider
            .complete(&Request {
                messages: vec![Message::user("hi")],
                ..Default::default()
            })
            .await
            .unwrap();

        assert_eq!(resp.content, "Hello!");
    }

    #[tokio::test]
    async fn complete_api_error() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&server)
            .await;

        let provider = AnthropicProvider::with_base_url("key", "model", server.uri());
        let err = provider.complete(&Request::default()).await.unwrap_err();
        match err {
            ProviderError::Api { status, message } => {
                assert_eq!(status, 429);
                assert!(message.contains("rate limited"));
            }
            other => panic!("expected Api error, got {other}"),
        }
    }

    #[tokio::test]
    async fn stream_success() {
        let server = MockServer::start().await;

        let sse_body = [
            "event: message_start\ndata: {\"type\":\"message_start\"}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\"}\n\n",
            "event: content_block_delta\ndata: {\"delta\":{\"text\":\"Hello\"}}\n\n",
            "event: content_block_delta\ndata: {\"delta\":{\"text\":\" world\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\"}\n\n",
            "event: message_delta\ndata: {\"usage\":{\"output_tokens\":5}}\n\n",
            "event: message_stop\ndata: {}\n\n",
        ]
        .join("");

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_string(sse_body))
            .mount(&server)
            .await;

        let provider = AnthropicProvider::with_base_url("key", "model", server.uri());
        let mut stream = provider
            .stream(&Request {
                messages: vec![Message::user("hi")],
                ..Default::default()
            })
            .await
            .unwrap();

        let mut text = String::new();
        let mut got_done = false;
        while let Some(chunk) = stream.next().await {
            match chunk {
                StreamChunk::Delta(t) => text.push_str(&t),
                StreamChunk::Thinking(_) => {}
                StreamChunk::Done { usage } => {
                    got_done = true;
                    let u = usage.unwrap();
                    assert_eq!(u.output_tokens, 5);
                }
                StreamChunk::Error(e) => panic!("unexpected error: {e}"),
            }
        }
        assert_eq!(text, "Hello world");
        assert!(got_done);
    }

    #[tokio::test]
    async fn stream_api_error() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(500).set_body_string("server error"))
            .mount(&server)
            .await;

        let provider = AnthropicProvider::with_base_url("key", "model", server.uri());
        match provider.stream(&Request::default()).await {
            Err(ProviderError::Api { status, .. }) => assert_eq!(status, 500),
            Err(other) => panic!("expected Api error, got {other}"),
            Ok(_) => panic!("expected error"),
        }
    }

    #[tokio::test]
    async fn stream_ends_without_stop() {
        let server = MockServer::start().await;

        // Stream with content but no message_stop/message_delta
        let sse_body = "event: content_block_delta\ndata: {\"delta\":{\"text\":\"hi\"}}\n\n";

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_string(sse_body))
            .mount(&server)
            .await;

        let provider = AnthropicProvider::with_base_url("key", "model", server.uri());
        let mut stream = provider
            .stream(&Request {
                messages: vec![Message::user("hi")],
                ..Default::default()
            })
            .await
            .unwrap();

        let mut chunks = Vec::new();
        while let Some(chunk) = stream.next().await {
            chunks.push(chunk);
        }

        // Should get Delta("hi") then Done { usage: None } from stream end
        assert!(chunks.len() >= 2);
        match &chunks[0] {
            StreamChunk::Delta(t) => assert_eq!(t, "hi"),
            other => panic!("expected Delta, got {other:?}"),
        }
        // Last chunk should be Done
        match chunks.last().unwrap() {
            StreamChunk::Done { usage } => assert!(usage.is_none()),
            other => panic!("expected Done, got {other:?}"),
        }
    }
}
