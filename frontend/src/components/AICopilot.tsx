import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import type {
  FormEvent,
} from "react";

import {
  Bot,
  BrainCircuit,
  Database,
  LoaderCircle,
  LockKeyhole,
  MessageCircle,
  Send,
  ShieldCheck,
  Sparkles,
  Wrench,
} from "lucide-react";

import ReactMarkdown
  from "react-markdown";

import remarkGfm
  from "remark-gfm";

import {
  askCopilot,
} from "../api/dashboard";

import type {
  ChatMessage,
} from "../types/dashboard";


// =========================================================
// Suggested questions
// =========================================================

const SUGGESTED_QUESTIONS = [
  "哪一種鋼材缺陷最常見？請告訴我數量和比例。",
  "K_Scatch 模型主要看哪些特徵？",
  "請列出模型 Confidence 最高的 5 筆預測樣本。",
  "Steel_Plate_Thickness 是 K_Scatch 的製造根因嗎？",
];


// =========================================================
// Helpers
// =========================================================

function createMessageId() {

  return `${Date.now()}-${Math.random()}`;
}


function getEvidenceLabel(
  toolsUsed: string[]
) {

  if (
    toolsUsed.includes(
      "get_defect_drivers"
    )
  ) {
    return "SHAP 分析";
  }

  if (
    toolsUsed.includes(
      "get_defect_distribution"
    )
  ) {
    return "品質資料分析";
  }

  if (
    toolsUsed.includes(
      "get_quality_overview"
    )
  ) {
    return "資料集總覽";
  }

  if (
    toolsUsed.includes(
      "get_high_confidence_predictions"
    )
  ) {
    return "模型推論結果";
  }

  return (
    toolsUsed.length > 0
      ? "工具證據"
      : "Policy / System Guidance"
  );
}


function getGuardrailLabel(
  message: ChatMessage
) {

  if (
    message.policyDecision
    ?.startsWith(
      "blocked_"
    )
  ) {
    return "Request Blocked";
  }

  if (
    message.content
      .toLowerCase()
      .includes(
        "root cause"
      )
    ||
    message.content
      .includes(
        "根因"
      )
    ||
    message.content
      .includes(
        "因果"
      )
  ) {
    return "Causality Protected";
  }

  return "Grounded Response";
}


// =========================================================
// Evidence badges
// =========================================================

function EvidenceBadges({
  message,
}: {
  message: ChatMessage;
}) {

  const tools =
    message.toolsUsed ?? [];

  const evidence =
    getEvidenceLabel(
      tools
    );

  const guardrail =
    getGuardrailLabel(
      message
    );

  return (
    <div className="copilot-evidence-row">

      {tools.length > 0 && (

        <div className="copilot-evidence-badge blue">

          <Wrench
            size={13}
          />

          <span>
            使用工具
          </span>

          <strong>
            {tools.join(
              ", "
            )}
          </strong>

        </div>
      )}


      <div className="copilot-evidence-badge teal">

        <Database
          size={13}
        />

        <span>
          證據來源
        </span>

        <strong>
          {evidence}
        </strong>

      </div>


      <div className="copilot-evidence-badge green">

        <ShieldCheck
          size={13}
        />

        <span>
          Guardrail
        </span>

        <strong>
          {guardrail}
        </strong>

      </div>

    </div>
  );
}


// =========================================================
// Markdown answer renderer
// =========================================================

function MarkdownAnswer({
  content,
}: {
  content: string;
}) {

  return (
    <div className="copilot-markdown">

      <ReactMarkdown
        remarkPlugins={[
          remarkGfm
        ]}
      >
        {content}
      </ReactMarkdown>

    </div>
  );
}


// =========================================================
// Main component
// =========================================================

export default function AICopilot() {

  const [
    question,
    setQuestion,
  ] = useState("");

  const [
    messages,
    setMessages,
  ] = useState<
    ChatMessage[]
  >([]);

  const [
    loading,
    setLoading,
  ] = useState(
    false
  );

  const [
    error,
    setError,
  ] = useState<
    string | null
  >(null);


  const conversationEndRef =
    useRef<HTMLDivElement | null>(
      null
    );


  const hasConversation =
    messages.length > 0;


  const assistantMessages =
    useMemo(
      () =>
        messages.filter(
          message =>
            message.role
            === "assistant"
        ),
      [
        messages
      ]
    );


  // =======================================================
  // Auto-scroll
  // =======================================================

  useEffect(
    () => {

      conversationEndRef
        .current
        ?.scrollIntoView({
          behavior:
            "smooth",

          block:
            "nearest",
        });

    },
    [
      messages,
      loading,
    ]
  );


  // =======================================================
  // Submit question
  // =======================================================

  async function submitQuestion(
    submittedQuestion?: string
  ) {

    const text =
      (
        submittedQuestion
        ?? question
      ).trim();

    if (
      !text
      || loading
    ) {
      return;
    }


    const userMessage:
      ChatMessage = {

        id:
          createMessageId(),

        role:
          "user",

        content:
          text,
      };


    setMessages(
      previous => [
        ...previous,
        userMessage,
      ]
    );

    setQuestion("");

    setLoading(
      true
    );

    setError(
      null
    );


    try {

      const response =
        await askCopilot(
          text
        );


      const assistantMessage:
        ChatMessage = {

        id:
          createMessageId(),

        role:
          "assistant",

        content:
          response.answer,

        model:
          response.model,

        toolsUsed:
          response.tools_used,

        policyDecision:
          response.policy_decision,
      };


      setMessages(
        previous => [
          ...previous,
          assistantMessage,
        ]
      );


    } catch (err) {

      console.error(
        err
      );

      setError(
        "AI Copilot 查詢失敗，請確認 Ollama 與 FastAPI 服務是否正常。"
      );

    } finally {

      setLoading(
        false
      );
    }
  }


  function handleSubmit(
    event: FormEvent
  ) {

    event.preventDefault();

    submitQuestion();
  }


  return (
    <section
      className="ai-copilot-section"
      id="ai-copilot"
    >

      {/* ============================================= */}
      {/* TITLE */}
      {/* ============================================= */}

      <div className="copilot-section-title">

        <div>

          <span className="section-eyebrow">
            AI Quality Copilot
          </span>

          <h2>
            具證據依據的製造品質 AI 助理
          </h2>

          <p>
            使用自然語言查詢鋼材品質分析、
            模型預測證據與 SHAP 可解釋性。
          </p>

        </div>


        <div className="copilot-engine-badge">

          <BrainCircuit
            size={15}
          />

          Qwen3.5 · Local LLM

        </div>

      </div>


      <div className="copilot-layout">

        {/* ============================================= */}
        {/* LEFT — SUGGESTIONS */}
        {/* ============================================= */}

        <aside className="copilot-suggestions">

          <div className="copilot-side-header">

            <MessageCircle
              size={18}
            />

            <div>

              <span>
                建議問題
              </span>

              <strong>
                探索鋼材品質資料
              </strong>

            </div>

          </div>


          <div className="suggestion-list">

            {SUGGESTED_QUESTIONS.map(
              (
                suggestion,
                index
              ) => (

                <button
                  className="suggestion-button"
                  key={suggestion}
                  onClick={
                    () =>
                      submitQuestion(
                        suggestion
                      )
                  }
                  disabled={
                    loading
                  }
                >

                  <div className="suggestion-number">
                    {String(
                      index + 1
                    ).padStart(
                      2,
                      "0"
                    )}
                  </div>

                  <span>
                    {suggestion}
                  </span>

                </button>

              )
            )}

          </div>


          <div className="copilot-safety-card">

            <LockKeyhole
              size={17}
            />

            <div>

              <strong>
                受控 AI 存取
              </strong>

              <p>
                僅允許使用預先定義的分析工具。
                禁止任意 SQL 與憑證存取；
                SHAP 預測因子不視為已確認的製造根因。
              </p>

            </div>

          </div>

        </aside>


        {/* ============================================= */}
        {/* RIGHT — CHAT */}
        {/* ============================================= */}

        <div className="copilot-chat-card">

          <div className="copilot-chat-header">

            <div className="copilot-avatar">

              <Bot
                size={20}
              />

            </div>

            <div>

              <strong>
                鋼材品質 AI Copilot
              </strong>

              <span>
                回答由 Allowlisted Tools 提供證據
              </span>

            </div>


            <div className="copilot-online">

              <span />

              已連線

            </div>

          </div>


          <div className="copilot-conversation">

            {!hasConversation && (

              <div className="copilot-empty-state">

                <div className="copilot-empty-icon">

                  <Sparkles
                    size={29}
                  />

                </div>

                <h3>
                  詢問鋼材品質問題
                </h3>

                <p>
                  AI Copilot 可查詢品質統計、
                  SHAP 模型預測因子，
                  並透過受控 Function Calling
                  取得資料後回答。
                </p>

              </div>
            )}


            {messages.map(
              message => (

                <div
                  key={
                    message.id
                  }
                  className={
                    message.role
                    === "user"
                      ? "chat-message user"
                      : "chat-message assistant"
                  }
                >

                  <div className="chat-message-label">

                    {message.role
                    === "user"
                      ? "你"
                      : "AI Copilot"
                    }

                    {message.role
                    === "assistant"
                    &&
                    message.model
                    &&
                    (
                      <span>
                        {message.model}
                      </span>
                    )
                    }

                  </div>


                  <div className="chat-message-body">

                    {message.role
                    === "assistant"
                      ? (
                        <MarkdownAnswer
                          content={
                            message.content
                          }
                        />
                      )
                      : (
                        message.content
                      )
                    }

                  </div>


                  {message.role
                  === "assistant"
                  &&
                  (
                    <EvidenceBadges
                      message={
                        message
                      }
                    />
                  )}

                </div>
              )
            )}


            {loading && (

              <div className="copilot-thinking">

                <LoaderCircle
                  className="spin"
                  size={17}
                />

                Qwen 正在選擇允許的 Tool，
                並取得證據後產生回答...

              </div>
            )}


            {error && (

              <div className="copilot-error">

                {error}

              </div>
            )}


            <div
              ref={
                conversationEndRef
              }
            />

          </div>


          <form
            className="copilot-input-area"
            onSubmit={
              handleSubmit
            }
          >

            <input
              value={
                question
              }
              onChange={
                event =>
                  setQuestion(
                    event
                      .target
                      .value
                  )
              }
              placeholder="詢問缺陷分布、模型表現、SHAP 預測因子..."
              disabled={
                loading
              }
            />

            <button
              type="submit"
              disabled={
                loading
                ||
                !question.trim()
              }
            >

              {loading
                ? (
                  <LoaderCircle
                    className="spin"
                    size={17}
                  />
                )
                : (
                  <Send
                    size={17}
                  />
                )
              }

              傳送

            </button>

          </form>


          <div className="copilot-footer">

            <span>
              {
                assistantMessages
                  .length
              } 筆 Grounded Response
            </span>

            <span>
              ·
            </span>

            <span>
              Local Inference
            </span>

            <span>
              ·
            </span>

            <span>
              Human-governed
            </span>

          </div>

        </div>

      </div>

    </section>
  );
}