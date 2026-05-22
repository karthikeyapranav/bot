import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import RNFS from 'react-native-fs';

let session: InferenceSession | null = null;

// Download from HuggingFace: Xenova/all-MiniLM-L6-v2 (ONNX, ~23MB)
export const loadEmbeddingModel = async () => {
  const modelPath = `${RNFS.DocumentDirectoryPath}/all-MiniLM-L6-v2.onnx`;
  session = await InferenceSession.create(modelPath);
};

export const getEmbedding = async (text: string): Promise<number[]> => {
  if (!session) throw new Error('Embedding model not loaded');
  // Tokenize (simplified — use a real tokenizer in production)
  // For a proper solution, bundle the tokenizer vocab and tokenize in JS
  const inputIds = tokenize(text); // returns number[]
  
  const feeds = {
    input_ids: new Tensor('int64', BigInt64Array.from(inputIds.map(BigInt)), [1, inputIds.length]),
    attention_mask: new Tensor('int64', BigInt64Array.from(inputIds.map(() => 1n)), [1, inputIds.length]),
  };
  const output = await session.run(feeds);
  // Mean pooling over token embeddings
  const embeddings = output['last_hidden_state'].data as Float32Array;
  return Array.from(embeddings.slice(0, 384)); // 384-dim
};