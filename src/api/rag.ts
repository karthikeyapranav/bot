import {getEmbedding} from './embeddingModel';
import {
  insertChunkWithEmbedding,
  retrieveTopK,
  type RetrievedChunk,
} from './vectorDb';

// Split text into overlapping chunks of ~150 words
export function chunkText(text: string, chunkSize = 150, overlap = 20): string[] {
  const words = text.split(/\s+/);
  const chunks: string[] = [];
  let i = 0;
  while (i < words.length) {
    const chunk = words.slice(i, i + chunkSize).join(' ');
    chunks.push(chunk);
    i += chunkSize - overlap; // slide forward with overlap
  }
  return chunks;
}

// Call this when user loads a document
export async function ingestText(
  text: string,
  onProgress?: (done: number, total: number) => void,
) {
  const chunks = chunkText(text);
  for (let i = 0; i < chunks.length; i++) {
    const embedding = await getEmbedding(chunks[i]);
    insertChunkWithEmbedding(chunks[i], embedding);
    onProgress?.(i + 1, chunks.length);
  }
  console.log(`Ingested ${chunks.length} chunks`);
}

// Build the RAG-enriched prompt to send to the LLM
export async function buildRAGPrompt(
  userQuestion: string,
): Promise<{prompt: string; chunks: RetrievedChunk[]}> {
  const queryEmbed = await getEmbedding(userQuestion);
  const topChunks = retrieveTopK(queryEmbed, 5);

  if (topChunks.length === 0) {
    // No context found — pass question directly
    return {prompt: userQuestion, chunks: []};
  }

  const context = topChunks
    .map((chunk, index) => {
      const source = chunk.source ? `Source: ${chunk.source}` : 'Source: unknown';
      const page = chunk.page != null ? `Page: ${chunk.page}` : 'Page: unknown';
      return `[Chunk ${index + 1} | ${source} | ${page}]\n${chunk.text}`;
    })
    .join('\n\n---\n\n');

  const prompt =
    `Use the following retrieved context to answer the question. ` +
    `Summarize the relevant facts clearly. ` +
    `If the context does not contain the answer, say so.\n\n` +
    `Context:\n${context}\n\n` +
    `Question: ${userQuestion}\n` +
    `Answer:`;

  return {prompt, chunks: topChunks};
}
