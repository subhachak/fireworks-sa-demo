import { useState, useEffect, useRef } from "react";

const DOCS = [
  {
    id: "doc_001",
    title: "AI Vendor Evaluation Policy",
    category: "Governance",
    content: `AI Vendor Evaluation Policy v2.1

PURPOSE: This policy establishes the framework for evaluating third-party AI vendors. All GenAI infrastructure decisions must follow this process before procurement approval.

APPROVAL THRESHOLDS:
- Proof of Concept: VP Engineering approval
- Production Pilot under $50K: Director approval plus Security review
- Enterprise Deployment over $50K: CTO and CISO approval required

SECURITY REQUIREMENTS: Vendors must hold SOC 2 Type II certification. No PII may be transmitted without a signed Data Processing Agreement. Vendors must confirm they do not use customer prompts for model training.`
  },
  {
    id: "doc_002",
    title: "LLM Inference Runbook",
    category: "Engineering",
    content: `LLM Inference Infrastructure Runbook — Platform Engineering

ARCHITECTURE:
- Primary: Fireworks AI serverless — Llama 3.3 70B
- Fallback: Azure OpenAI GPT-4o — only when latency exceeds 2 seconds
- Embedding: Fireworks nomic-embed-text-v1.5

LATENCY TARGETS:
- p50 under 300ms for standard completions
- p99 under 800ms per Fireworks SLA
- First token under 100ms for streaming

INCIDENT RESPONSE:
- P1 above 5 percent error rate: page on-call immediately
- P2 latency above 2x SLA: monitor 15 min then trigger fallback
- P3 quality regression: log to dashboard, review weekly`
  },
  {
    id: "doc_003",
    title: "GenAI Intake Process",
    category: "Program Management",
    content: `GenAI Use Case Intake — AI Center of Excellence

INTAKE REQUIREMENTS: Teams must provide business problem with quantified impact, data availability and PII classification, success metrics and KPIs, named business owner, and target timeline.

PRIORITIZATION SCORING (1 to 5):
- Strategic Alignment: 30 percent weight
- Business Impact: 30 percent weight
- Technical Feasibility: 25 percent weight
- Data Readiness: 15 percent weight

THRESHOLDS:
- Score 4.0 and above: fast-tracked to 2-week discovery sprint
- Score 3.0 to 3.9: queued for next quarterly planning
- Score below 3.0: deferred with written feedback`
  },
  {
    id: "doc_004",
    title: "Model Selection Guidelines",
    category: "Engineering",
    content: `Model Selection Guidelines — Enterprise AI Architecture

RECOMMENDED MODELS:

High complexity reasoning tasks like legal analysis and architecture review:
Use Llama 3.1 405B or GPT-4o. Latency 1 to 3 seconds. Cost $3 to $5 per million tokens.

Standard Q&A and summarization — PREFERRED DEFAULT:
Use Llama 3.3 70B via Fireworks. Latency 300 to 600ms. Cost $0.90 per million tokens. Four times cheaper than GPT-4o with comparable quality.

High volume real-time tasks like classification and routing:
Use Llama 3.1 8B. Latency under 150ms. Cost $0.20 per million tokens.

COMPLIANCE: For HIPAA or PCI data always use Fireworks dedicated deployments, never shared serverless infrastructure.`
  },
  {
    id: "doc_005",
    title: "RAG Architecture Best Practices",
    category: "Engineering",
    content: `RAG Architecture Best Practices — Platform Engineering

CHUNKING: Default 512 tokens with 50-token overlap. Never split mid-sentence. Use sentence boundary detection. Long documents use 1024 tokens with 100-token overlap.

RETRIEVAL: Top-K of 5 for standard queries, 10 for complex multi-hop. Cosine similarity threshold of 0.75 minimum. Hybrid BM25 plus dense retrieval improves precision by 15 percent. Cross-encoder reranking improves top-1 accuracy by 20 percent.

PROMPT RULES: System prompt must instruct the model to answer only from provided context, cite source documents for every claim, say I don't have information on that if context is insufficient, and never hallucinate statistics.

EVALUATION TARGETS:
- Faithfulness above 95 percent
- Answer relevancy above 0.85
- End to end latency under 800ms`
  }
];

function tokenize(text) {
  return text.toLowerCase().replace(/[^a-z0-9\s]/g, "").split(/\s+/).filter(Boolean);
}

function buildChunks(docs) {
  const chunks = [];
  docs.forEach(doc => {
    const sentences = doc.content.split(/(?<=[.!?])\s+/);
    let current = [], size = 0;
    sentences.forEach(s => {
      const words = s.split(" ");
      if (size + words.length > 120 && current.length) {
        chunks.push({
          docId: doc.id,
          docTitle: doc.title,
          category: doc.category,
          text: current.join(" ")
        });
        current = current.slice(-10);
        size = current.join(" ").split(" ").length;
      }
      current.push(s);
      size += words.length;
    });
    if (current.length) {
      chunks.push({
        docId: doc.id,
        docTitle: doc.title,
        category: doc.category,
        text: current.join(" ")
      });
    }
  });
  return chunks;
}

function tfidfScore(queryTokens, chunkText) {
  const chunkTokens = tokenize(chunkText);
  const freq = {};
  chunkTokens.forEach(t => { freq[t] = (freq[t] || 0) + 1; });
  let score = 0;
  queryTokens.forEach(qt => {
    if (freq[qt]) score += (1 + Math.log(freq[qt]));
  });
  return score / (Math.sqrt(chunkTokens.length) + 1);
}

function retrieve(query, chunks, topK = 4) {
  const qTokens = tokenize(query);
  return chunks
    .map(c => ({ ...c, score: tfidfScore(qTokens, c.text) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .filter(c => c.score > 0);
}

async function* streamFromFireworks(messages, apiKey) {
  const res = await fetch("/inference/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": "Bearer " + apiKey,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: "accounts/fireworks/models/deepseek-v3p2",
      messages,
      max_tokens: 1024,
      temperature: 0.1,
      stream: true,
      top_p: 0.9
    })
  });

  if (!res.ok) throw new Error("Fireworks API error: " + res.status);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop();
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const d = line.slice(6).trim();
        if (d === "[DONE]") return;
        try {
          const j = JSON.parse(d);
          const tok = j.choices?.[0]?.delta?.content;
          if (tok) yield tok;
        } catch {}
      }
    }
  }
}

async function* mockStream(sources) {
  const text = sources.length
    ? "Based on the knowledge base, here is what I found:\n\n" +
      sources[0].text.slice(0, 400) + "...\n\n" +
      "Please refer to the source documents below for complete details."
    : "I don't have information on that in the current knowledge base.";

  for (const word of text.split(" ")) {
    await new Promise(r => setTimeout(r, 25));
    yield word + " ";
  }
}

export default function App() {
  const [apiKey, setApiKey] = useState("");
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const [chunks, setChunks] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [showConfig, setShowConfig] = useState(false);
  const bottomRef = useRef(null);

  // Build chunks on load
  useEffect(() => {
    setChunks(buildChunks(DOCS));
  }, []);

  // Auto scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleQuery = async () => {
    const text = query.trim();
    if (!text || isStreaming) return;
    setQuery("");

    // Add user message
    const userMsg = { id: Date.now(), role: "user", content: text };
    setMessages(prev => [...prev, userMsg]);
    setIsStreaming(true);

    // Retrieve relevant chunks
    const retrieved = retrieve(text, chunks);

    // Add empty assistant message to fill in
    const assistantId = Date.now() + 1;
    setMessages(prev => [...prev, {
      id: assistantId,
      role: "assistant",
      content: "",
      streaming: true,
      sources: retrieved
    }]);

    // Build prompt
    const context = retrieved.map((r, i) =>
      "[Source " + (i + 1) + ": " + r.docTitle + "]\n" + r.text
    ).join("\n\n---\n\n");

    const messages_payload = [
      {
        role: "system",
        content: "You are an enterprise AI assistant. Answer ONLY from the provided context. Cite sources as [Source N]. If context is insufficient say: I don't have information on that in the current knowledge base."
      },
      {
        role: "user",
        content: "Context:\n\n" + (context || "No relevant documents found.") + "\n\n---\nQuestion: " + text
      }
    ];

    // Stream response
    let fullContent = "";
    try {
      const stream = apiKey.trim()
        ? streamFromFireworks(messages_payload, apiKey.trim())
        : mockStream(retrieved);

      for await (const tok of stream) {
        fullContent += tok;
        setMessages(prev => prev.map(m =>
          m.id === assistantId ? { ...m, content: fullContent } : m
        ));
      }
    } catch (err) {
      fullContent = "⚠ Error: " + err.message;
      setMessages(prev => prev.map(m =>
        m.id === assistantId ? { ...m, content: fullContent } : m
      ));
    }

    setMessages(prev => prev.map(m =>
      m.id === assistantId ? { ...m, streaming: false } : m
    ));
    setIsStreaming(false);
  };

  return (
    <div style={{
      minHeight: "100vh",
      background: "#020817",
      color: "#e2e8f0",
      fontFamily: "monospace",
      display: "flex",
      flexDirection: "column"
    }}>

      {/* Header */}
      <div style={{
        borderBottom: "1px solid #1e293b",
        padding: "0 24px",
        height: "56px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        flexShrink: 0
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div style={{
            width: "28px", height: "28px", borderRadius: "6px",
            background: "linear-gradient(135deg, #f97316, #fb923c)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontWeight: "bold", color: "white", fontSize: "13px"
          }}>F</div>
          <span style={{ fontWeight: 600, fontSize: "13px" }}>FIREWORKS AI</span>
          <span style={{ color: "#334155" }}>/</span>
          <span style={{ color: "#64748b", fontSize: "12px" }}>Enterprise Knowledge Assistant</span>
        </div>
        <button onClick={() => setShowConfig(!showConfig)} style={{
          background: "transparent", border: "1px solid #1e293b",
          color: "#94a3b8", padding: "6px 12px", borderRadius: "6px",
          cursor: "pointer", fontSize: "12px"
        }}>⚙ API Key</button>
      </div>

      {/* Config drawer */}
      {showConfig && (
        <div style={{
          background: "#0f172a", borderBottom: "1px solid #1e293b",
          padding: "12px 24px", display: "flex", alignItems: "center", gap: "12px"
        }}>
          <span style={{ color: "#64748b", fontSize: "12px" }}>FIREWORKS_API_KEY</span>
          <input
            type="password"
            value={apiKey}
            onChange={e => setApiKey(e.target.value)}
            placeholder="fw_..."
            style={{
              background: "#020817", border: "1px solid #1e293b",
              color: "#e2e8f0", padding: "8px 12px", borderRadius: "6px",
              fontFamily: "monospace", fontSize: "12px", width: "360px"
            }}
          />
          <span style={{
            fontSize: "11px", padding: "4px 10px", borderRadius: "6px",
            background: apiKey ? "#10b98122" : "#f9731622",
            color: apiKey ? "#10b981" : "#f97316",
            border: "1px solid " + (apiKey ? "#10b98144" : "#f9731644")
          }}>
            {apiKey ? "✓ Live mode" : "Demo mode"}
          </span>
        </div>
      )}

      {/* Messages */}
      <div style={{ flex: 1, overflow: "auto", padding: "24px" }}>
        {messages.length === 0 && (
          <div style={{ color: "#334155", fontSize: "13px", textAlign: "center", marginTop: "60px" }}>
            Ask anything about your AI policies, runbooks, or guidelines.
          </div>
        )}

        {messages.map(msg => (
          <div key={msg.id} style={{
            display: "flex",
            justifyContent: msg.role === "user" ? "flex-end" : "flex-start",
            marginBottom: "20px"
          }}>
            <div style={{ maxWidth: "75%" }}>

              {/* Bubble */}
              <div style={{
                background: msg.role === "user" ? "#1e3a5f" : "#0f172a",
                border: "1px solid " + (msg.role === "user" ? "#1d4ed844" : "#1e293b"),
                borderRadius: msg.role === "user" ? "16px 4px 16px 16px" : "4px 16px 16px 16px",
                padding: "12px 16px",
                fontSize: "14px",
                lineHeight: 1.7,
                whiteSpace: "pre-wrap"
              }}>
                {msg.content}
                {msg.streaming && (
                  <span style={{
                    display: "inline-block", width: "2px", height: "14px",
                    background: "#f97316", marginLeft: "2px", verticalAlign: "middle",
                    animation: "blink 0.8s infinite"
                  }} />
                )}
              </div>

              {/* Sources */}
              {!msg.streaming && msg.sources && msg.sources.length > 0 && (
                <div style={{ marginTop: "10px" }}>
                  <div style={{ color: "#475569", fontSize: "11px", marginBottom: "6px" }}>
                    SOURCES ({msg.sources.length})
                  </div>
                  {msg.sources.map((s, i) => (
                    <div key={i} style={{
                      background: "#0f172a",
                      border: "1px solid #1e293b",
                      borderLeft: "3px solid #f97316",
                      borderRadius: "6px",
                      padding: "8px 12px",
                      marginBottom: "6px",
                      fontSize: "12px"
                    }}>
                      <span style={{ color: "#f97316", fontWeight: 600 }}>[{i + 1}]</span>
                      {" "}<span style={{ color: "#e2e8f0" }}>{s.docTitle}</span>
                      <span style={{ color: "#475569" }}> · {s.category}</span>
                    </div>
                  ))}
                </div>
              )}

            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div style={{
        borderTop: "1px solid #1e293b",
        padding: "16px 24px",
        background: "#020817"
      }}>
        <div style={{
          display: "flex", gap: "12px", alignItems: "flex-end",
          background: "#0f172a", border: "1px solid #1e293b",
          borderRadius: "12px", padding: "12px 16px"
        }}>
          <textarea
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleQuery();
              }
            }}
            placeholder="Ask about your AI policies, runbooks, or guidelines..."
            disabled={isStreaming}
            rows={1}
            style={{
              flex: 1, background: "transparent", border: "none",
              color: "#e2e8f0", fontSize: "14px", fontFamily: "monospace",
              resize: "none", lineHeight: 1.6,
              opacity: isStreaming ? 0.5 : 1
            }}
          />
          <button
            onClick={handleQuery}
            disabled={!query.trim() || isStreaming}
            style={{
              background: isStreaming ? "#1e293b" : "linear-gradient(135deg, #f97316, #fb923c)",
              border: "none", borderRadius: "8px",
              width: "36px", height: "36px",
              cursor: isStreaming ? "not-allowed" : "pointer",
              color: "white", fontSize: "16px",
              opacity: (!query.trim() || isStreaming) ? 0.4 : 1
            }}
          >↑</button>
        </div>
        <div style={{ marginTop: "8px", textAlign: "center", color: "#334155", fontSize: "10px" }}>
          Retrieval via TF-IDF · Generation via Fireworks AI llama-v3p3-70b-instruct · Enter to send
        </div>
      </div>

      <style>{`
        @keyframes blink { 0%,100% { opacity:1 } 50% { opacity:0 } }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 2px; }
        textarea:focus, input:focus { outline: none; }
      `}</style>

    </div>
  );
}