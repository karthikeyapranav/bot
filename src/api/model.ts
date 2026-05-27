import RNFS from 'react-native-fs';

export const GEMMA_MODEL = {
  name: 'gemma-3-1b-Q2_K.gguf',
  url: 'https://drive.usercontent.google.com/download?id=17Qo0qpaKVHSb83gHOSsnlt3gEh799CjW&export=download&authuser=0&confirm=t',
  displayName: 'Gemma 3 1B · Q2_K',
  sizeHint: '~500 MB',
};

export const downloadModel = async (
  modelName: string,
  modelUrl: string,
  onProgress: (progress: number) => void,
): Promise<string> => {
  const destPath = `${RNFS.DocumentDirectoryPath}/${modelName}`;

  if (!modelName || !modelUrl) {
    throw new Error('Invalid model name or URL');
  }

  const fileExists = await RNFS.exists(destPath);
  if (fileExists) {
    const stat = await RNFS.stat(destPath);
    if (stat.size > 10_000_000) {
      console.log(`Model already exists at ${destPath}, skipping download.`);
      onProgress(100);
      return destPath;
    } else {
      // Corrupted / partial — delete and re-download
      console.warn('Existing file too small, deleting and re-downloading...');
      await RNFS.unlink(destPath);
    }
  }

  console.log('Starting download from:', modelUrl);

  const downloadResult = await RNFS.downloadFile({
    fromUrl: modelUrl,
    toFile: destPath,
    progressDivider: 2,
    headers: {
      'User-Agent': 'Mozilla/5.0',
    },
    begin: res => {
      console.log('Download started, content-length:', res.contentLength);
    },
    progress: ({
      bytesWritten,
      contentLength,
    }: {
      bytesWritten: number;
      contentLength: number;
    }) => {
      if (contentLength > 0) {
        const pct = Math.floor((bytesWritten / contentLength) * 100);
        onProgress(pct);
      }
    },
  }).promise;

  if (downloadResult.statusCode === 200) {
    // ── Sanity check: make sure we got the actual model, not a Drive HTML page ──
    const stat = await RNFS.stat(destPath);
    if (stat.size < 10_000_000) {
      await RNFS.unlink(destPath);
      throw new Error(
        'Downloaded file is too small (~' +
          Math.round(stat.size / 1024) +
          ' KB).\n\n' +
          'Google Drive likely returned a virus-scan warning page instead of the model.\n\n' +
          'Fix: Upload the GGUF to HuggingFace (free) and update the URL in model.ts.',
      );
    }
    return destPath;
  } else {
    const partialExists = await RNFS.exists(destPath);
    if (partialExists) await RNFS.unlink(destPath);
    throw new Error(`Download failed with HTTP status: ${downloadResult.statusCode}`);
  }
};

export const isModelDownloaded = async (): Promise<boolean> => {
  const destPath = `${RNFS.DocumentDirectoryPath}/${GEMMA_MODEL.name}`;
  const exists = await RNFS.exists(destPath);
  if (!exists) return false;
  // Also verify it's a real model file, not a stale HTML page
  const stat = await RNFS.stat(destPath);
  return stat.size > 10_000_000;
};