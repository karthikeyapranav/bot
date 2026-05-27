import {getEmbedding} from './embeddingModel';
import {retrieveTopK, type RetrievedChunk} from './vectorDb';

// ── Constants ─────────────────────────────────────────────────────────────────
const TOP_K = 3;
const MAX_CHUNK_WORDS = 150;

// Distance threshold — sqlite-vec returns L2 distance.
// Empirically, distance > 1.0 means the chunk is probably unrelated.
// Tune this if your DB uses cosine (lower threshold ~0.3).
const RELEVANCE_THRESHOLD = 1.0;

// ── Agriculture topic keywords ────────────────────────────────────────────────
// If the user question contains NONE of these, skip RAG entirely and let
// the LLM answer from its own knowledge.
const AGRICULTURE_KEYWORDS = [
  'crop', 'crops', 'farm', 'farming', 'farmer', 'soil', 'fertilizer',
  'fertiliser', 'pesticide', 'irrigation', 'harvest', 'seed', 'seeds',
  'plant', 'plants', 'cultivation', 'cultivate', 'paddy', 'rice', 'wheat',
  'maize', 'corn', 'sorghum', 'millet', 'pulse', 'lentil', 'chickpea',
  'vegetable', 'fruit', 'orchard', 'greenhouse', 'compost', 'manure',
  'weed', 'pest', 'disease', 'blight', 'fungus', 'insect', 'drought',
  'rainfall', 'season', 'rabi', 'kharif', 'zaid', 'yield', 'acres',
  'hectare', 'field', 'agriculture', 'agricultural', 'agri', 'horticulture',
  'floriculture', 'sericulture', 'aquaculture', 'livestock', 'poultry',
  'cattle', 'dairy', 'goat', 'sheep', 'pH', 'nitrogen', 'phosphorus',
  'potassium', 'NPK', 'urea', 'spray', 'sowing', 'transplant', 'pruning',
  'tilling', 'ploughing', 'mulching', 'drip', 'sprinkler',
];

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
 * Checks if the question is agriculture-related by keyword matching.
 * Simple but fast — no embedding call needed for this check.
 */
function isAgricultureQuestion(question: string): boolean {
  const lower = question.toLowerCase();
  return AGRICULTURE_KEYWORDS.some(kw => lower.includes(kw));
}

/**
 * Build the prompt to send to the LLM.
 *
 * Decision logic:
 *
 *  1. If NOT agriculture → skip RAG, return plain question.
 *     LLM answers from its own knowledge with no context block.
 *
 *  2. If agriculture + chunks retrieved with good distance → RAG prompt.
 *     LLM uses context to answer.
 *
 *  3. If agriculture + chunks retrieved but ALL have high distance (weak) →
 *     Tell LLM the context is weak, ask it to mention that and supplement
 *     with its own knowledge.
 *
 *  4. If agriculture + zero chunks → plain question, LLM answers itself.
 */
export async function buildRAGPrompt(
  userQuestion: string,
): Promise<RAGPromptResult> {

  // ── Step 1: Topic check ───────────────────────────────────────────────────
  if (!isAgricultureQuestion(userQuestion)) {
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
    // Weak retrieval — tell LLM the context may not be perfect
    prompt =
      `You are an agricultural expert. The following passages may be relevant.\n\n` +
      `${contextBlock}\n\n` +
      `Answer this question using the passages above AND your own agricultural knowledge. ` +
      `Do not ask the user for more information. Give a direct answer.\n\n` +
      `Respond in the same language as the question.\n` +
      `Question: ${userQuestion}\n` +
      `Answer:`;
  } else {
    // Good retrieval
    prompt =
      `You are an agricultural expert. The following passages are from an agriculture document.\n\n` +
      `${contextBlock}\n\n` +
      `Using ONLY the passages above, answer this question in 2-3 sentences. ` +
      `Do not ask for more information. If the passages don't contain the answer, say "I don't have that information in my knowledge base."\n\n` +
      `Respond in the same language as the question.\n` +
      `Question: ${userQuestion}\n` +
      `Answer:`;
  }

  return {
    prompt,
    chunks: truncatedChunks,
    decision: isWeak ? 'rag_weak' : 'rag_good',
  };
}