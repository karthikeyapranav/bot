import * as ort from 'onnxruntime-react-native';
import RNFS from 'react-native-fs';

let session: ort.InferenceSession | null = null;
let vocab: Map<string, number> = new Map();

const MODEL_ASSET = 'model-int8.onnx';
const VOCAB_ASSET = 'vocab.txt';
const MODEL_DEST = `${RNFS.DocumentDirectoryPath}/model-int8.onnx`;
const VOCAB_DEST = `${RNFS.DocumentDirectoryPath}/vocab.txt`;

// ── Copy from assets to filesystem on first run ──────────────────────────────
async function copyAssetIfNeeded(asset: string, dest: string) {
  const exists = await RNFS.exists(dest);
  if (!exists) {
    await RNFS.copyFileAssets(asset, dest);
    console.log(`Copied ${asset} → ${dest}`);
  }
}

// ── Tiny WordPiece tokenizer (no dependency needed for BERT vocab) ───────────
async function loadVocab() {
  const raw = await RNFS.readFile(VOCAB_DEST, 'utf8');
  const lines = raw.split('\n');
  vocab = new Map();
  lines.forEach((token, idx) => {
    const t = token.trim();
    if (t) vocab.set(t, idx);
  });
}

function tokenize(text: string, maxLen = 128): {
  input_ids: number[];
  attention_mask: number[];
  token_type_ids: number[];
} {
  // Basic BERT WordPiece tokenize (handles ##subwords)
  const CLS = vocab.get('[CLS]') ?? 101;
  const SEP = vocab.get('[SEP]') ?? 102;
  const UNK = vocab.get('[UNK]') ?? 100;
  const PAD = 0;

  const words = text.toLowerCase().trim().split(/\s+/);
  const tokens: number[] = [CLS];

  for (const word of words) {
    if (tokens.length >= maxLen - 1) break;

    // Try full word first
    if (vocab.has(word)) {
      tokens.push(vocab.get(word)!);
      continue;
    }

    // WordPiece: greedily split into subwords
    let remaining = word;
    let isFirst = true;
    let wordTokens: number[] = [];
    let failed = false;

    while (remaining.length > 0) {
      let found = false;
      for (let end = remaining.length; end > 0; end--) {
        const sub = isFirst ? remaining.slice(0, end) : '##' + remaining.slice(0, end);
        if (vocab.has(sub)) {
          wordTokens.push(vocab.get(sub)!);
          remaining = remaining.slice(end);
          isFirst = false;
          found = true;
          break;
        }
      }
      if (!found) {
        wordTokens = [UNK];
        failed = true;
        break;
      }
    }
    tokens.push(...wordTokens);
    if (failed) continue;
  }

  tokens.push(SEP);

  // Pad to maxLen
  const paddedLen = Math.min(tokens.length, maxLen);
  const input_ids = tokens.slice(0, paddedLen);
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
  await copyAssetIfNeeded(MODEL_ASSET, MODEL_DEST);
  await copyAssetIfNeeded(VOCAB_ASSET, VOCAB_DEST);
  await loadVocab();

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
  const hiddenState = output['last_hidden_state'].data as Float32Array;
  const hiddenSize = 384;

  return meanPool(hiddenState, attention_mask, maxLen, hiddenSize);
}