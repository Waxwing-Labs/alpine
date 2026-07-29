use std::time::Duration;

use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// ModelId
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId(pub String);

impl ModelId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for ModelId {
    fn default() -> Self {
        Self("default".into())
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ---------------------------------------------------------------------------
// Message
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

impl Message {
    /// Construct a message from a role and explicit content blocks.
    pub fn new(role: Role, content: Vec<ContentBlock>) -> Self {
        Self { role, content }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::text(content)],
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentBlock::text(content)],
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: vec![ContentBlock::text(content)],
        }
    }

    /// A user message carrying a single tool result (the conventional way to
    /// return tool output to the model).
    pub fn tool_result(
        tool_use_id: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: tool_use_id.into(),
                content: content.into(),
                is_error,
            }],
        }
    }

    /// Concatenate all text blocks into a single string. Non-text blocks
    /// (tool_use / tool_result) are ignored. This is the convenience accessor
    /// for callers that only care about textual content.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// All `tool_use` blocks in this message.
    pub fn tool_uses(&self) -> impl Iterator<Item = &ContentBlock> {
        self.content
            .iter()
            .filter(|b| matches!(b, ContentBlock::ToolUse { .. }))
    }
}

// ---------------------------------------------------------------------------
// Content blocks
// ---------------------------------------------------------------------------

/// A single piece of a message's content. Messages are sequences of blocks so
/// that a turn can mix text with tool-use requests (assistant) and tool
/// results (user) — the shape required for provider tool/function calling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    /// The model is requesting a tool invocation.
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// The caller is returning the result of a tool invocation.
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
}

impl ContentBlock {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

// ---------------------------------------------------------------------------
// Request / Response
// ---------------------------------------------------------------------------

/// Extended-thinking (reasoning) configuration. Honored by providers that
/// support it (Anthropic); ignored by those that don't (Ollama).
#[derive(Debug, Clone)]
pub enum ThinkingConfig {
    /// Adaptive extended thinking (Claude 4.6+ / 5). The model decides how much
    /// to reason; bound overall spend with `effort` (Anthropic's
    /// `output_config.effort`) so it leaves room in `max_tokens` for a visible
    /// answer. `display = Summarized` streams readable reasoning summaries
    /// (surfaced as [`StreamChunk::Thinking`]); `Omitted` (Anthropic's default)
    /// yields empty thinking text. Anthropic disallows a custom `temperature`
    /// alongside thinking.
    Adaptive { display: ThinkingDisplay, effort: Option<Effort> },
    /// Explicitly disable extended thinking (direct answer).
    Disabled,
    /// Legacy fixed-budget thinking. Pre-4.6 models only — 400s on Claude 4.7+/5
    /// (which removed `budget_tokens` in favor of adaptive thinking + effort).
    Enabled { budget_tokens: u32 },
}

/// Whether a provider returns readable reasoning summaries or hides them.
#[derive(Debug, Clone, Copy)]
pub enum ThinkingDisplay {
    /// Stream a human-readable summary of the reasoning.
    Summarized,
    /// Omit reasoning text (Anthropic's default) — thinking still happens.
    Omitted,
}

/// Reasoning-effort hint, mapped to Anthropic's `output_config.effort`. Lower
/// effort means less time spent reasoning (and lower cost/latency).
#[derive(Debug, Clone, Copy)]
pub enum Effort {
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

#[derive(Debug, Clone, Default)]
pub struct Request {
    pub messages: Vec<Message>,
    pub model: ModelId,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub system: Option<String>,
    pub stop: Vec<String>,
    /// Tools the model may call. Empty (the default) means no tool calling —
    /// identical behavior to before tool support existed.
    pub tools: Vec<ToolDefinition>,
    /// Extended-thinking control. `None` (default) sends no `thinking` field,
    /// preserving each provider's default behavior.
    pub thinking: Option<ThinkingConfig>,
}

#[derive(Debug, Clone)]
pub struct Response {
    /// Flattened text content (all text blocks concatenated).
    pub content: String,
    /// Any tool calls the model requested this turn. Empty when the model
    /// returned only text.
    pub tool_calls: Vec<ToolUse>,
    pub usage: Usage,
    pub model: ModelId,
    pub finish_reason: FinishReason,
    pub latency: Duration,
    pub raw: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

/// A provider-agnostic tool the model may call. `input_schema` is a JSON Schema
/// describing the tool's arguments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

impl ToolDefinition {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
        }
    }
}

/// A tool call the model requested.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolUse {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum FinishReason {
    #[default]
    Stop,
    MaxTokens,
    ContentFilter,
    /// The model stopped because it wants to call one or more tools.
    ToolUse,
    Other(String),
}

impl Response {
    /// Reconstruct the assistant turn this response represents as a `Message`,
    /// preserving both text and any tool-use requests. Append this to the
    /// conversation before returning tool results, so the provider sees the
    /// request that the results answer (Anthropic in particular requires the
    /// assistant `tool_use` to immediately precede the user `tool_result`).
    pub fn assistant_message(&self) -> Message {
        let mut content = Vec::new();
        if !self.content.is_empty() {
            content.push(ContentBlock::text(&self.content));
        }
        for call in &self.tool_calls {
            content.push(ContentBlock::ToolUse {
                id: call.id.clone(),
                name: call.name.clone(),
                input: call.input.clone(),
            });
        }
        Message {
            role: Role::Assistant,
            content,
        }
    }
}

impl std::fmt::Display for Response {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "[{}] ({:.0?})", self.model, self.latency)?;
        writeln!(f, "{}", self.content)?;
        write!(
            f,
            "tokens: {} in / {} out | finish: {:?}",
            self.usage.input_tokens, self.usage.output_tokens, self.finish_reason
        )
    }
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum StreamChunk {
    /// A chunk of the model's visible answer text.
    Delta(String),
    /// A chunk of the model's *reasoning* (extended-thinking) text. Emitted only
    /// by providers/models that stream a thinking block. Kept distinct from
    /// [`StreamChunk::Delta`] so a host can render it separately (a live
    /// "thinking…" view) and exclude it from the persisted answer.
    Thinking(String),
    Done { usage: Option<Usage> },
    Error(String),
}

/// Convenience alias used throughout the crate.
pub type StreamResponse<'a> = BoxStream<'a, StreamChunk>;

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct EmbedRequest {
    pub model: ModelId,
    pub input: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Embedding {
    pub vectors: Vec<Vec<f32>>,
    pub model: ModelId,
    pub usage: Usage,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::time::Duration;

    // -- ModelId ---------------------------------------------------------------

    #[test]
    fn model_id_new_and_as_str() {
        let m = ModelId::new("gpt-4");
        assert_eq!(m.as_str(), "gpt-4");
    }

    #[test]
    fn model_id_default() {
        assert_eq!(ModelId::default().as_str(), "default");
    }

    #[test]
    fn model_id_display() {
        let m = ModelId::new("claude-3");
        assert_eq!(format!("{m}"), "claude-3");
    }

    #[test]
    fn model_id_eq_and_hash() {
        let a = ModelId::new("x");
        let b = ModelId::new("x");
        let c = ModelId::new("y");
        assert_eq!(a, b);
        assert_ne!(a, c);

        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn model_id_serde_roundtrip() {
        let m = ModelId::new("llama3.2");
        let json = serde_json::to_string(&m).unwrap();
        let back: ModelId = serde_json::from_str(&json).unwrap();
        assert_eq!(m, back);
    }

    // -- Message ---------------------------------------------------------------

    #[test]
    fn message_user() {
        let m = Message::user("hi");
        assert_eq!(m.role, Role::User);
        assert_eq!(m.text(), "hi");
    }

    #[test]
    fn message_assistant() {
        let m = Message::assistant("ok");
        assert_eq!(m.role, Role::Assistant);
        assert_eq!(m.text(), "ok");
    }

    #[test]
    fn message_system() {
        let m = Message::system("you are helpful");
        assert_eq!(m.role, Role::System);
        assert_eq!(m.text(), "you are helpful");
    }

    #[test]
    fn message_text_concatenates_and_ignores_non_text() {
        let m = Message::new(
            Role::Assistant,
            vec![
                ContentBlock::text("a"),
                ContentBlock::ToolUse {
                    id: "t1".into(),
                    name: "x".into(),
                    input: serde_json::json!({}),
                },
                ContentBlock::text("b"),
            ],
        );
        assert_eq!(m.text(), "ab");
        assert_eq!(m.tool_uses().count(), 1);
    }

    #[test]
    fn message_tool_result_helper() {
        let m = Message::tool_result("tu_1", "result body", false);
        assert_eq!(m.role, Role::User);
        match &m.content[0] {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                assert_eq!(tool_use_id, "tu_1");
                assert_eq!(content, "result body");
                assert!(!is_error);
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn content_block_serde_roundtrip() {
        for block in [
            ContentBlock::text("hi"),
            ContentBlock::ToolUse {
                id: "id".into(),
                name: "search".into(),
                input: serde_json::json!({"q": "x"}),
            },
            ContentBlock::ToolResult {
                tool_use_id: "id".into(),
                content: "ok".into(),
                is_error: false,
            },
        ] {
            let json = serde_json::to_string(&block).unwrap();
            let back: ContentBlock = serde_json::from_str(&json).unwrap();
            assert_eq!(block, back);
        }
    }

    // -- Role ------------------------------------------------------------------

    #[test]
    fn role_serde_roundtrip() {
        for (role, expected) in [
            (Role::User, "\"user\""),
            (Role::Assistant, "\"assistant\""),
            (Role::System, "\"system\""),
        ] {
            let json = serde_json::to_string(&role).unwrap();
            assert_eq!(json, expected);
            let back: Role = serde_json::from_str(&json).unwrap();
            assert_eq!(back, role);
        }
    }

    // -- Request ---------------------------------------------------------------

    #[test]
    fn request_default() {
        let r = Request::default();
        assert!(r.messages.is_empty());
        assert_eq!(r.model, ModelId::default());
        assert!(r.max_tokens.is_none());
        assert!(r.temperature.is_none());
        assert!(r.system.is_none());
        assert!(r.stop.is_empty());
    }

    // -- Response Display ------------------------------------------------------

    #[test]
    fn response_display() {
        let resp = Response {
            content: "Hello!".into(),
            tool_calls: vec![],
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
            model: ModelId::new("test-model"),
            finish_reason: FinishReason::Stop,
            latency: Duration::from_millis(1234),
            raw: serde_json::Value::Null,
        };
        let s = format!("{resp}");
        assert!(s.contains("test-model"));
        assert!(s.contains("Hello!"));
        assert!(s.contains("10 in"));
        assert!(s.contains("5 out"));
        assert!(s.contains("Stop"));
        // latency formatted with {:.0?} — should contain "1.234s" or "1234ms"
        assert!(s.contains("1"));
    }

    // -- FinishReason ----------------------------------------------------------

    #[test]
    fn finish_reason_default_is_stop() {
        assert_eq!(FinishReason::default(), FinishReason::Stop);
    }

    #[test]
    fn finish_reason_variants() {
        assert_eq!(FinishReason::Stop, FinishReason::Stop);
        assert_ne!(FinishReason::Stop, FinishReason::MaxTokens);
        assert_ne!(FinishReason::MaxTokens, FinishReason::ContentFilter);
        let other = FinishReason::Other("custom".into());
        assert_eq!(other, FinishReason::Other("custom".into()));
        assert_ne!(other, FinishReason::Other("different".into()));
    }

    // -- Usage -----------------------------------------------------------------

    #[test]
    fn usage_default() {
        let u = Usage::default();
        assert_eq!(u.input_tokens, 0);
        assert_eq!(u.output_tokens, 0);
    }

    // -- StreamChunk -----------------------------------------------------------

    #[test]
    fn stream_chunk_debug() {
        let _ = format!("{:?}", StreamChunk::Delta("hi".into()));
        let _ = format!("{:?}", StreamChunk::Done { usage: None });
        let _ = format!(
            "{:?}",
            StreamChunk::Done {
                usage: Some(Usage::default())
            }
        );
        let _ = format!("{:?}", StreamChunk::Error("err".into()));
    }

    // -- EmbedRequest / Embedding ----------------------------------------------

    #[test]
    fn embed_request_construction() {
        let r = EmbedRequest {
            model: ModelId::new("nomic"),
            input: vec!["hello".into(), "world".into()],
        };
        assert_eq!(r.model.as_str(), "nomic");
        assert_eq!(r.input.len(), 2);
    }

    #[test]
    fn embedding_construction() {
        let e = Embedding {
            vectors: vec![vec![0.1, 0.2], vec![0.3, 0.4]],
            model: ModelId::new("nomic"),
            usage: Usage {
                input_tokens: 4,
                output_tokens: 0,
            },
        };
        assert_eq!(e.vectors.len(), 2);
        assert_eq!(e.vectors[0].len(), 2);
        assert_eq!(e.model.as_str(), "nomic");
    }
}
