import * as ort from 'onnxruntime-react-native';
import {TokenizerLoader} from '@lenml/tokenizers';
import RNFS from 'react-native-fs';

let session: ort.InferenceSession | null = null;
let tokenizer: any = null;

const MODEL_ASSET = 'model-int8.onnx';
const TOKENIZER_ASSET = 'tokenizer.json';
const MODEL_DEST = `${RNFS.DocumentDirectoryPath}/model-int8.onnx`;
const TOKENIZER_DEST = `${RNFS.DocumentDirectoryPath}/tokenizer.json`;

// ── Copy from assets to filesystem so app storage matches bundled assets ─────
async function copyAsset(asset: string, dest: string) {
  const exists = await RNFS.exists(dest);
  if (exists) await RNFS.unlink(dest);
  await RNFS.copyFileAssets(asset, dest);
  console.log(`Copied ${asset} → ${dest}`);
}

async function loadTokenizer() {
  const raw = await RNFS.readFile(TOKENIZER_DEST, 'utf8');
  tokenizer = TokenizerLoader.fromPreTrained({
    tokenizerJSON: JSON.parse(raw),
    tokenizerConfig: {},
  });
}

function tokenize(text: string, maxLen = 128): {
  input_ids: number[];
  attention_mask: number[];
  token_type_ids: number[];
} {
  if (!tokenizer) throw new Error('Tokenizer not loaded.');

  const tokens = tokenizer
    .encode(text, {add_special_tokens: true})
    .slice(0, maxLen);
  const PAD = 0;
  const input_ids = tokens.slice(0, maxLen);
  const attention_mask = input_ids.map(() => 1);
  const token_type_ids = input_ids.map(() => 0);

  // Pad
  while (input_ids.length < maxLen) {
    input_ids.push(PAD);
    attention_mask.push(0);
    token_type_ids.push(0);
  }

  return {input_ids, attention_mask, token_type_ids};
}

// ── Mean pooling over last_hidden_state with attention mask ──────────────────
function meanPool(
  hiddenState: Float32Array,
  attentionMask: number[],
  seqLen: number,
  hiddenSize: number,
): number[] {
  const result = new Array(hiddenSize).fill(0);
  let count = 0;

  for (let t = 0; t < seqLen; t++) {
    if (attentionMask[t] === 0) continue;
    count++;
    for (let h = 0; h < hiddenSize; h++) {
      result[h] += hiddenState[t * hiddenSize + h];
    }
  }

  for (let h = 0; h < hiddenSize; h++) {
    result[h] /= Math.max(count, 1);
  }

  // L2 normalise (cosine similarity needs this)
  const norm = Math.sqrt(result.reduce((s, v) => s + v * v, 0));
  return result.map(v => v / Math.max(norm, 1e-9));
}

// ── Public API ────────────────────────────────────────────────────────────────
export async function loadEmbeddingModel() {
  console.log('Loading embedding model...');
  await copyAsset(MODEL_ASSET, MODEL_DEST);
  await copyAsset(TOKENIZER_ASSET, TOKENIZER_DEST);
  await loadTokenizer();

  session = await ort.InferenceSession.create(MODEL_DEST, {
    executionProviders: ['cpu'],
  });

  console.log('Embedding model ready. Input names:', session.inputNames);
  console.log('Output names:', session.outputNames);
}

export async function getEmbedding(text: string): Promise<number[]> {
  if (!session) throw new Error('Embedding model not loaded. Call loadEmbeddingModel() first.');

  const maxLen = 128;
  const {input_ids, attention_mask, token_type_ids} = tokenize(text, maxLen);

  const feeds: Record<string, ort.Tensor> = {
    input_ids: new ort.Tensor(
      'int64',
      BigInt64Array.from(input_ids.map(BigInt)),
      [1, maxLen],
    ),
    attention_mask: new ort.Tensor(
      'int64',
      BigInt64Array.from(attention_mask.map(BigInt)),
      [1, maxLen],
    ),
    token_type_ids: new ort.Tensor(
      'int64',
      BigInt64Array.from(token_type_ids.map(BigInt)),
      [1, maxLen],
    ),
  };

  const output = await session.run(feeds);

  // Output is last_hidden_state: [1, seqLen, 384]
  const hiddenState = output.last_hidden_state.data as Float32Array;
  const hiddenSize = 384;

  return meanPool(hiddenState, attention_mask, maxLen, hiddenSize);
}
