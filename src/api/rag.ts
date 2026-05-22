import {getEmbedding} from './embeddingModel';
import {insertChunkWithEmbedding, retrieveTopK} from './vectorDb';

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
export async function buildRAGPrompt(userQuestion: string): Promise<string> {
  const queryEmbed = await getEmbedding(userQuestion);
  const topChunks = retrieveTopK(queryEmbed, 3);

  if (topChunks.length === 0) {
    // No context found — pass question directly
    return userQuestion;
  }

  const context = topChunks.join('\n\n---\n\n');
  return (
    `Use the following retrieved context to answer the question. ` +
    `If the context does not contain the answer, say so.\n\n` +
    `Context:\n${context}\n\n` +
    `Question: ${userQuestion}\n` +
    `Answer:`
  );
}