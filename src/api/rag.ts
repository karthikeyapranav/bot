import {getEmbedding} from './embeddingModel';
import {retrieveTopK, type RetrievedChunk} from './vectorDb';

// ── Constants ─────────────────────────────────────────────────────────────────
const TOP_K = 3;
const MAX_CHUNK_WORDS = 150;

// Distance threshold — sqlite-vec returns L2 distance.
// Empirically, distance > 1.0 means the chunk is probably unrelated.
const RELEVANCE_THRESHOLD = 1.0;

// ── LLM Completion callback type ─────────────────────────────────────────────
// We accept a thin wrapper so rag.ts stays decoupled from llama.rn internals.
// app.tsx will pass `(messages, onToken) => Promise<string>`.
export type LLMCompletionFn = (
  messages: {role: 'system' | 'user' | 'assistant'; content: string}[],
) => Promise<string>;

export type RAGDecision =
  | 'rag_good'        // question is agri + chunks are relevant
  | 'rag_weak'        // question is agri + chunks found but distance is high
  | 'rag_no_chunks'   // question is agri + zero chunks returned
  | 'skip_not_agri';  // question is NOT agriculture-related → pure LLM

export type RAGPromptResult = {
  prompt: string;
  chunks: RetrievedChunk[];
  decision: RAGDecision;
};

// ─────────────────────────────────────────────────────────────────────────────

function truncateChunk(text: string, maxWords = MAX_CHUNK_WORDS): string {
  const words = text.trim().split(/\s+/);
  if (words.length <= maxWords) return text.trim();
  return words.slice(0, maxWords).join(' ') + '…';
}

/**
 * Uses the LLM itself to decide if the question is agriculture-related.
 *
 * We give the model a tiny, deterministic classification prompt and ask
 * it to reply with exactly one word: YES or NO.
 *
 * Falls back to `true` (assume agriculture) on any error so RAG is
 * attempted rather than silently skipped.
 */
async function isAgricultureQuestion(
  question: string,
  llmComplete: LLMCompletionFn,
): Promise<boolean> {
  // Keep prompt as short as possible — 1B models drift on long instructions.
  // Single-line system + minimal user message = most reliable YES/NO output.
  const classifyPrompt = `Agriculture related? YES or NO.\nQuestion: ${question}\nAnswer:`;

  try {
    const reply = await llmComplete([
      {role: 'system', content: 'You classify questions. Reply YES or NO only.'},
      {role: 'user',   content: classifyPrompt},
    ]);

    const cleaned = reply.trim().toUpperCase();
    console.log('[RAG classifier]', JSON.stringify(question), '→', cleaned);

    // Accept Y/YES/Yes — also treat empty/garbage reply as YES (safe fallback)
    if (cleaned === '' || cleaned === 'N' || cleaned.startsWith('NO')) {
      return false;
    }
    return true; // YES, Yes, Y, or any unexpected output → try RAG
  } catch (err) {
    console.warn('[RAG classifier] LLM call failed, defaulting to YES:', err);
    return true;
  }
}

/**
 * Build the prompt to send to the LLM.
 *
 * Decision logic (now driven by LLM classifier, not keywords):
 *
 *  1. LLM says NOT agriculture → skip RAG, return plain question.
 *
 *  2. Agriculture + good chunks (distance ≤ threshold) → RAG prompt using
 *     retrieved context only.
 *
 *  3. Agriculture + weak chunks (distance > threshold) → RAG prompt that
 *     tells the LLM context is uncertain; supplement with own knowledge.
 *
 *  4. Agriculture + zero chunks → plain question, LLM answers itself.
 */
export async function buildRAGPrompt(
  userQuestion: string,
  llmComplete: LLMCompletionFn,
): Promise<RAGPromptResult> {

  // ── Step 1: LLM topic classification ─────────────────────────────────────
  const isAgri = await isAgricultureQuestion(userQuestion, llmComplete);

  if (!isAgri) {
    return {
      prompt: userQuestion,
      chunks: [],
      decision: 'skip_not_agri',
    };
  }

  // ── Step 2: Embed + retrieve ──────────────────────────────────────────────
  const queryEmbed = await getEmbedding(userQuestion);
  const topChunks = retrieveTopK(queryEmbed, TOP_K);

  if (topChunks.length === 0) {
    return {
      prompt: userQuestion,
      chunks: [],
      decision: 'rag_no_chunks',
    };
  }

  // ── Step 3: Truncate chunks ───────────────────────────────────────────────
  const truncatedChunks = topChunks.map(chunk => ({
    ...chunk,
    text: truncateChunk(chunk.text),
  }));

  // ── Step 4: Relevance check ───────────────────────────────────────────────
  const bestDistance = Math.min(...truncatedChunks.map(c => Number(c.distance)));
  const isWeak = bestDistance > RELEVANCE_THRESHOLD;

  // ── Step 5: Build prompt ──────────────────────────────────────────────────
  const contextBlock = truncatedChunks
    .map((chunk, i) => {
      const src = chunk.source ? `Source: ${chunk.source}` : 'Source: unknown';
      const pg = chunk.page != null ? ` | Page: ${chunk.page}` : '';
      const dist = Number.isFinite(Number(chunk.distance))
        ? ` | dist: ${Number(chunk.distance).toFixed(3)}`
        : '';
      return `[Chunk ${i + 1} | ${src}${pg}${dist}]\n${chunk.text}`;
    })
    .join('\n\n');

  let prompt: string;

  if (isWeak) {
    prompt =
      `మీరు వ్యవసాయ నిపుణుడు. క్రింది సమాచారం సంబంధితంగా ఉండవచ్చు.\n\n` +
      `${contextBlock}\n\n` +
      `పై సమాచారాన్ని మరియు మీ స్వంత వ్యవసాయ జ్ఞానాన్ని ఉపయోగించి సమాధానం ఇవ్వండి. ` +
      `తెలుగులో సమాధానం ఇవ్వండి.\n` +
      `ప్రశ్న: ${userQuestion}\n` +
      `సమాధానం:`;
  } else {
    prompt =
      `మీరు వ్యవసాయ నిపుణుడు. క్రింది సమాచారం వ్యవసాయ పత్రం నుండి తీసుకోబడింది.\n\n` +
      `${contextBlock}\n\n` +
      `పై సమాచారాన్ని మాత్రమే ఉపయోగించి 2-3 వాక్యాలలో సమాధానం ఇవ్వండి. ` +
      `తెలుగులో సమాధానం ఇవ్వండి.\n` +
      `ప్రశ్న: ${userQuestion}\n` +
      `సమాధానం:`;
  }

  return {
    prompt,
    chunks: truncatedChunks,
    decision: isWeak ? 'rag_weak' : 'rag_good',
  };
}