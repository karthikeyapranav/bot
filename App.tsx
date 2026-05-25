import React, {useState, useEffect, useRef} from 'react';
import {
  StyleSheet,
  View,
  Alert,
  SafeAreaView,
  ScrollView,
  Text,
  TouchableOpacity,
  ActivityIndicator,
  TextInput,
  Platform,
  KeyboardAvoidingView,
} from 'react-native';

import axios from 'axios';
import RNFS from 'react-native-fs';
import {initLlama, releaseAllLlama} from 'llama.rn';

import {downloadModel} from './src/api/model';
import ProgressBar from './src/components/ProgressBar';

// ── RAG imports ───────────────────────────────────────────────────────────────
import {loadEmbeddingModel} from './src/api/embeddingModel';
import {
  initVectorDB,
  type RetrievedChunk,
} from './src/api/vectorDb';
import {buildRAGPrompt} from './src/api/rag';

// ─────────────────────────────────────────────────────────────────────────────

type Message = {
  role: 'system' | 'user' | 'assistant';
  content: string;         // raw content sent to LLM
  displayContent?: string; // what the user sees (original question, not the RAG prompt)
  ragChunks?: RetrievedChunk[];
};

type Page = 'modelSelection' | 'conversation' | 'ragSetup';

const SYSTEM_MESSAGE: Message = {
  role: 'system',
  content:
    'You are a helpful assistant. When context is provided, use it to answer accurately. If the context does not contain the answer, say so honestly.',
};

const HF_TO_GGUF: Record<string, string> = {
  'Llama-3.2-1B-Instruct': 'medmekk/Llama-3.2-1B-Instruct.GGUF',
  'DeepSeek-R1-Distill-Qwen-1.5B': 'medmekk/DeepSeek-R1-Distill-Qwen-1.5B.GGUF',
  'Qwen2-0.5B-Instruct': 'medmekk/Qwen2.5-0.5B-Instruct.GGUF',
  'SmolLM2-1.7B-Instruct': 'medmekk/SmolLM2-1.7B-Instruct.GGUF',
};

const MODEL_FORMATS = Object.keys(HF_TO_GGUF);

const STOP_WORDS = [
  '</s>',
  '<|end|>',
  'user:',
  'assistant:',
  '<|im_end|>',
  '<|eot_id|>',
  '<|end▁of▁sentence|>',
  '<｜end▁of▁sentence｜>',
];

// ─────────────────────────────────────────────────────────────────────────────

function App(): React.JSX.Element {
  // ── LLM state ──────────────────────────────────────────────────────────────
  const [conversation, setConversation] = useState<Message[]>([SYSTEM_MESSAGE]);
  const [selectedModelFormat, setSelectedModelFormat] = useState('');
  const [selectedGGUF, setSelectedGGUF] = useState<string | null>(null);
  const [availableGGUFs, setAvailableGGUFs] = useState<string[]>([]);
  const [userInput, setUserInput] = useState('');
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [llmContext, setLlmContext] = useState<any>(null);
  const [isDownloading, setIsDownloading] = useState(false);
  const [isFetching, setIsFetching] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);

  // ── RAG state ──────────────────────────────────────────────────────────────
  const [embeddingReady, setEmbeddingReady] = useState(false);
  const [embeddingLoading, setEmbeddingLoading] = useState(false);
  const [ragEnabled, setRagEnabled] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);
  const [chunksLoaded, setChunksLoaded] = useState(false);
  const [knowledgeChunkCount, setKnowledgeChunkCount] = useState(0);

  // ── Navigation ─────────────────────────────────────────────────────────────
  const [currentPage, setCurrentPage] = useState<Page>('modelSelection');

  const scrollRef = useRef<ScrollView>(null);

  // ── On mount: init vector DB + load embedding model ───────────────────────
  useEffect(() => {
    const setup = async () => {
      try {
        const chunkCount = await initVectorDB();
        setKnowledgeChunkCount(chunkCount);
        setChunksLoaded(chunkCount > 0);
        setRagEnabled(chunkCount > 0);
        console.log('Knowledge DB initialised');
      } catch (e) {
        console.warn('Knowledge DB init failed:', e);
      }

      setEmbeddingLoading(true);
      try {
        await loadEmbeddingModel();
        setEmbeddingReady(true);
        console.log('Embedding model ready');
      } catch (e) {
        console.warn('Embedding model load failed:', e);
        Alert.alert(
          'Embedding Warning',
          'Could not load embedding model. RAG will be disabled.\n\n' +
            'Make sure model-int8.onnx and vocab.txt are in android/app/src/main/assets/',
        );
      } finally {
        setEmbeddingLoading(false);
      }
    };
    setup();
  }, []);

  // Auto-scroll when new messages arrive
  useEffect(() => {
    setTimeout(() => scrollRef.current?.scrollToEnd({animated: true}), 100);
  }, [conversation]);

  // ── LLM helpers ────────────────────────────────────────────────────────────
  const handleFormatSelection = (format: string) => {
    setSelectedModelFormat(format);
    setAvailableGGUFs([]);
    fetchAvailableGGUFs(format);
  };

  const fetchAvailableGGUFs = async (modelFormat: string) => {
    setIsFetching(true);
    try {
      const repoPath = HF_TO_GGUF[modelFormat];
      if (!repoPath) throw new Error(`No repo mapping for: ${modelFormat}`);
      const response = await axios.get(
        `https://huggingface.co/api/models/${repoPath}`,
      );
      if (!response.data?.siblings) throw new Error('Invalid API response');
      const files = response.data.siblings
        .filter((f: {rfilename: string}) => f.rfilename.endsWith('.gguf'))
        .map((f: {rfilename: string}) => f.rfilename);
      setAvailableGGUFs(files);
    } catch (error) {
      Alert.alert('Error', error instanceof Error ? error.message : 'Fetch failed');
      setAvailableGGUFs([]);
    } finally {
      setIsFetching(false);
    }
  };

  const handleGGUFSelection = (file: string) => {
    setSelectedGGUF(file);
    Alert.alert(
      'Confirm Download',
      `Download ${file}?`,
      [
        {text: 'No', onPress: () => setSelectedGGUF(null), style: 'cancel'},
        {text: 'Yes', onPress: () => handleDownloadAndLoad(file)},
      ],
      {cancelable: false},
    );
  };

  const handleDownloadAndLoad = async (file: string) => {
    const downloadUrl = `https://huggingface.co/${HF_TO_GGUF[selectedModelFormat]}/resolve/main/${file}`;
    setIsDownloading(true);
    setDownloadProgress(0);
    try {
      const destPath = await downloadModel(file, downloadUrl, p =>
        setDownloadProgress(p),
      );
      Alert.alert('Downloaded', `Saved to: ${destPath}`);
      if (destPath) {
        const loaded = await loadLLM(file);
        if (loaded) setCurrentPage('conversation');
      }
    } catch (error) {
      Alert.alert('Download Error', error instanceof Error ? error.message : 'Unknown error');
    } finally {
      setIsDownloading(false);
    }
  };

  const loadLLM = async (modelName: string): Promise<boolean> => {
    try {
      const destPath = `${RNFS.DocumentDirectoryPath}/${modelName}`;
      const exists = await RNFS.exists(destPath);
      if (!exists) {
        Alert.alert('Error', 'Model file not found.');
        return false;
      }
      if (llmContext) {
        await releaseAllLlama();
        setLlmContext(null);
        setConversation([SYSTEM_MESSAGE]);
      }
      const ctx = await initLlama({
        model: destPath,
        use_mlock: true,
        n_ctx: 2048,
        n_gpu_layers: 1,
      });
      setLlmContext(ctx);
      Alert.alert('Model Loaded', 'LLM is ready to chat!');
      return true;
    } catch (error) {
      Alert.alert('Load Error', error instanceof Error ? error.message : 'Unknown');
      return false;
    }
  };

  // ── Send message (with optional RAG) ──────────────────────────────────────
  const handleSendMessage = async () => {
    if (!llmContext) {
      Alert.alert('No Model', 'Please download and load a model first.');
      return;
    }
    if (!userInput.trim()) return;

    const userText = userInput.trim();
    setUserInput('');
    setIsGenerating(true);

    try {
      let promptContent = userText;
      let retrievedChunks: RetrievedChunk[] = [];

      // If RAG is enabled and chunks are loaded, enrich the prompt
      if (ragEnabled && chunksLoaded && embeddingReady) {
        try {
          const ragResult = await buildRAGPrompt(userText);
          promptContent = ragResult.prompt;
          retrievedChunks = ragResult.chunks;
        } catch (ragErr) {
          console.warn('RAG prompt failed, using plain question:', ragErr);
          promptContent = userText;
          retrievedChunks = [];
        }
      }

      // Add the user turn — show original text to user, send enriched prompt to LLM
      const newConversation: Message[] = [
        ...conversation,
        {
          role: 'user',
          content: promptContent,       // this goes to the LLM
          displayContent: userText,     // this is shown in the chat bubble
          ragChunks: retrievedChunks,
        },
      ];
      setConversation(newConversation);

      // Build the messages array for the LLM (use content, not displayContent)
      const llmMessages = newConversation.map(m => ({
        role: m.role,
        content: m.content,
      }));

      const result = await llmContext.completion(
        {
          messages: llmMessages,
          n_predict: 1024,
          stop: STOP_WORDS,
        },
        (_data: {token: string}) => {
          // partial token callback — could stream here
        },
      );

      if (result?.text) {
        setConversation(prev => [
          ...prev,
          {role: 'assistant', content: result.text.trim()},
        ]);
      } else {
        throw new Error('Empty response from model.');
      }
    } catch (error) {
      Alert.alert(
        'Inference Error',
        error instanceof Error ? error.message : 'Unknown error.',
      );
    } finally {
      setIsGenerating(false);
    }
  };

  // ── RAG helpers ────────────────────────────────────────────────────────────
  const handleIngestDocument = async () => {
    if (!embeddingReady) {
      Alert.alert('Not Ready', 'Embedding model is still loading.');
      return;
    }
    setIsIngesting(true);
    try {
      const chunkCount = await initVectorDB();
      setKnowledgeChunkCount(chunkCount);
      setChunksLoaded(chunkCount > 0);
      setRagEnabled(chunkCount > 0);
      Alert.alert('Done!', `Knowledge DB loaded with ${chunkCount} chunks.`);
    } catch (e) {
      Alert.alert('DB Load Error', e instanceof Error ? e.message : 'Unknown');
    } finally {
      setIsIngesting(false);
    }
  };

  // ─────────────────────────────────────────────────────────────────────────
  // RENDER
  // ─────────────────────────────────────────────────────────────────────────

  return (
    <SafeAreaView style={styles.root}>
      <KeyboardAvoidingView
        style={styles.flex}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={0}>

        {/* ── Header ─────────────────────────────────────────────────────── */}
        <View style={styles.header}>
          <Text style={styles.headerTitle}>🦙 LlamaRAG</Text>
          <View style={styles.headerRight}>
            {/* RAG setup button — only show when a model is loaded */}
            {currentPage === 'conversation' && (
              <TouchableOpacity
                style={styles.headerBtn}
                onPress={() => setCurrentPage('ragSetup')}>
                <Text style={styles.headerBtnText}>
                  📄 {chunksLoaded ? 'Doc ✓' : 'Doc'}
                </Text>
              </TouchableOpacity>
            )}
            {currentPage === 'ragSetup' && (
              <TouchableOpacity
                style={styles.headerBtn}
                onPress={() => setCurrentPage('conversation')}>
                <Text style={styles.headerBtnText}>← Chat</Text>
              </TouchableOpacity>
            )}
          </View>
        </View>

        {/* ── Embedding status bar ────────────────────────────────────────── */}
        <View
          style={[
            styles.statusBar,
            embeddingReady ? styles.statusReady : styles.statusLoading,
          ]}>
          {embeddingLoading ? (
            <>
              <ActivityIndicator size="small" color="#fff" style={{marginRight: 8}} />
              <Text style={styles.statusText}>Loading embedding model…</Text>
            </>
          ) : embeddingReady ? (
            <Text style={styles.statusText}>
              ✓ Embedding ready
              {chunksLoaded
                ? `  |  ✓ DB loaded (${knowledgeChunkCount} chunks)`
                : '  |  No DB loaded'}
              {chunksLoaded
                ? ragEnabled
                  ? '  |  RAG ON'
                  : '  |  RAG OFF'
                : ''}
            </Text>
          ) : (
            <Text style={styles.statusText}>⚠ Embedding model failed to load</Text>
          )}
        </View>

        {/* ═══════════════════════════════════════════════════════════════════
            PAGE: MODEL SELECTION
        ═══════════════════════════════════════════════════════════════════ */}
        {currentPage === 'modelSelection' && !isDownloading && (
          <ScrollView contentContainerStyle={styles.scrollContent}>
            <Text style={styles.pageTitle}>Choose a Model</Text>

            <View style={styles.card}>
              <Text style={styles.sectionLabel}>Model Format</Text>
              {MODEL_FORMATS.map(format => (
                <TouchableOpacity
                  key={format}
                  style={[
                    styles.optionBtn,
                    selectedModelFormat === format && styles.optionBtnSelected,
                  ]}
                  onPress={() => handleFormatSelection(format)}>
                  <Text
                    style={[
                      styles.optionBtnText,
                      selectedModelFormat === format && styles.optionBtnTextSelected,
                    ]}>
                    {format}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            {selectedModelFormat !== '' && (
              <View style={styles.card}>
                <Text style={styles.sectionLabel}>Select GGUF Quantisation</Text>
                {isFetching && (
                  <ActivityIndicator
                    size="small"
                    color="#3B82F6"
                    style={{marginVertical: 12}}
                  />
                )}
                {availableGGUFs.map((file, idx) => (
                  <TouchableOpacity
                    key={idx}
                    style={[
                      styles.optionBtn,
                      selectedGGUF === file && styles.optionBtnSelected,
                    ]}
                    onPress={() => handleGGUFSelection(file)}>
                    <Text
                      style={[
                        styles.optionBtnTextSmall,
                        selectedGGUF === file && styles.optionBtnTextSelected,
                      ]}>
                      {file}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            )}
          </ScrollView>
        )}

        {/* ═══════════════════════════════════════════════════════════════════
            PAGE: DOWNLOADING
        ═══════════════════════════════════════════════════════════════════ */}
        {isDownloading && (
          <View style={styles.centeredCard}>
            <Text style={styles.sectionLabel}>Downloading</Text>
            <Text style={styles.smallLabel}>{selectedGGUF}</Text>
            <ProgressBar progress={downloadProgress} />
            <Text style={styles.progressPct}>
              {Math.round(downloadProgress * 100)}%
            </Text>
          </View>
        )}

        {/* ═══════════════════════════════════════════════════════════════════
            PAGE: RAG SETUP
        ═══════════════════════════════════════════════════════════════════ */}
        {currentPage === 'ragSetup' && (
          <ScrollView contentContainerStyle={styles.scrollContent}>
            <Text style={styles.pageTitle}>Document Setup</Text>

            {/* RAG toggle */}
            <View style={styles.card}>
              <Text style={styles.sectionLabel}>RAG Mode</Text>
              <TouchableOpacity
                style={[
                  styles.toggleBtn,
                  ragEnabled ? styles.toggleBtnOn : styles.toggleBtnOff,
                  (!chunksLoaded || !embeddingReady) && styles.toggleBtnDisabled,
                ]}
                disabled={!chunksLoaded || !embeddingReady}
                onPress={() => setRagEnabled(prev => !prev)}>
                <Text style={styles.toggleBtnText}>
                  {ragEnabled ? '✓ RAG Enabled' : 'RAG Disabled'}
                </Text>
              </TouchableOpacity>
              {!chunksLoaded && (
                <Text style={styles.hintText}>
                  Load the bundled knowledge DB below to enable RAG.
                </Text>
              )}
            </View>

            {/* Bundled knowledge DB */}
            <View style={styles.card}>
              <Text style={styles.sectionLabel}>Knowledge Database</Text>
              <Text style={styles.hintText}>
                Bundled asset: knowledge.db
                {knowledgeChunkCount
                  ? `\nIndexed chunks available: ${knowledgeChunkCount}`
                  : '\nNo chunks loaded yet.'}
              </Text>
              {isIngesting ? (
                <View style={styles.ingestProgress}>
                  <ActivityIndicator size="small" color="#3B82F6" />
                  <Text style={styles.ingestProgressText}>
                    Loading knowledge DB…
                  </Text>
                </View>
              ) : (
                <TouchableOpacity
                  style={[
                    styles.primaryBtn,
                    !embeddingReady && styles.primaryBtnDisabled,
                  ]}
                  disabled={!embeddingReady}
                  onPress={handleIngestDocument}>
                  <Text style={styles.primaryBtnText}>Reload Knowledge DB</Text>
                </TouchableOpacity>
              )}
            </View>

            {/* Refresh */}
            {chunksLoaded && (
              <View style={styles.card}>
                <Text style={styles.sectionLabel}>Manage Index</Text>
                <TouchableOpacity
                  style={styles.primaryBtn}
                  onPress={handleIngestDocument}>
                  <Text style={styles.primaryBtnText}>Reload Bundled DB</Text>
                </TouchableOpacity>
              </View>
            )}
          </ScrollView>
        )}

        {/* ═══════════════════════════════════════════════════════════════════
            PAGE: CONVERSATION
        ═══════════════════════════════════════════════════════════════════ */}
        {currentPage === 'conversation' && !isDownloading && (
          <>
            <ScrollView
              ref={scrollRef}
              style={styles.chatScroll}
              contentContainerStyle={styles.chatContent}>

              {/* RAG active badge */}
              {ragEnabled && chunksLoaded && (
                <View style={styles.ragBadge}>
                  <Text style={styles.ragBadgeText}>
                    📄 RAG active — answers grounded in your document
                  </Text>
                </View>
              )}

              {!llmContext && (
                <View style={styles.emptyState}>
                  <Text style={styles.emptyStateText}>
                    ← Go back to select and download a model to start chatting.
                  </Text>
                </View>
              )}

              {conversation.slice(1).map((msg, idx) => (
                <View
                  key={idx}
                  style={[
                    styles.bubbleWrapper,
                    msg.role === 'user'
                      ? styles.bubbleWrapperUser
                      : styles.bubbleWrapperAssistant,
                  ]}>
                  {msg.role === 'assistant' && (
                    <Text style={styles.roleLabel}>🦙 Llama</Text>
                  )}
                  <View
                    style={[
                      styles.bubble,
                      msg.role === 'user'
                        ? styles.bubbleUser
                        : styles.bubbleAssistant,
                    ]}>
                    <Text
                      style={[
                        styles.bubbleText,
                        msg.role === 'user' && styles.bubbleTextUser,
                      ]}>
                      {/* Show displayContent (original question) for user; full text for assistant */}
                      {msg.role === 'user'
                        ? msg.displayContent ?? msg.content
                        : msg.content}
                    </Text>
                  </View>
                  {msg.role === 'user' && msg.ragChunks?.length ? (
                    <View style={styles.retrievedPanel}>
                      <Text style={styles.retrievedTitle}>
                        Top {msg.ragChunks.length} retrieved chunks
                      </Text>
                      {msg.ragChunks.map((chunk, chunkIndex) => (
                        <View key={`${chunk.id}-${chunkIndex}`} style={styles.chunkCard}>
                          <Text style={styles.chunkMeta}>
                            #{chunkIndex + 1} · chunk {chunk.id}
                            {chunk.page != null ? ` · page ${chunk.page}` : ''}
                            {Number.isFinite(Number(chunk.distance))
                              ? ` · distance ${Number(chunk.distance).toFixed(4)}`
                              : ''}
                          </Text>
                          <Text style={styles.chunkText}>{chunk.text}</Text>
                          {chunk.source ? (
                            <Text style={styles.chunkSource}>{chunk.source}</Text>
                          ) : null}
                        </View>
                      ))}
                    </View>
                  ) : null}
                </View>
              ))}

              {isGenerating && (
                <View style={[styles.bubbleWrapper, styles.bubbleWrapperAssistant]}>
                  <Text style={styles.roleLabel}>🦙 Llama</Text>
                  <View style={[styles.bubble, styles.bubbleAssistant]}>
                    <ActivityIndicator size="small" color="#3B82F6" />
                  </View>
                </View>
              )}
            </ScrollView>

            {/* ── Input bar ─────────────────────────────────────────────── */}
            <View style={styles.inputBar}>
              {/* RAG quick toggle */}
              {embeddingReady && chunksLoaded && (
                <TouchableOpacity
                  style={[
                    styles.ragToggleChip,
                    ragEnabled ? styles.ragToggleChipOn : styles.ragToggleChipOff,
                  ]}
                  onPress={() => setRagEnabled(p => !p)}>
                  <Text style={styles.ragToggleChipText}>
                    {ragEnabled ? '📄 RAG' : '💬 RAG'}
                  </Text>
                </TouchableOpacity>
              )}

              <TextInput
                style={styles.textInput}
                placeholder={
                  llmContext
                    ? ragEnabled
                      ? 'Ask about your document…'
                      : 'Type a message…'
                    : 'Load a model first…'
                }
                placeholderTextColor="#94A3B8"
                value={userInput}
                onChangeText={setUserInput}
                multiline
                editable={!!llmContext && !isGenerating}
                onSubmitEditing={handleSendMessage}
                blurOnSubmit={false}
              />

              <TouchableOpacity
                style={[
                  styles.sendBtn,
                  (!llmContext || isGenerating || !userInput.trim()) &&
                    styles.sendBtnDisabled,
                ]}
                onPress={handleSendMessage}
                disabled={!llmContext || isGenerating || !userInput.trim()}>
                <Text style={styles.sendBtnText}>
                  {isGenerating ? '…' : '↑'}
                </Text>
              </TouchableOpacity>
            </View>
          </>
        )}

        {/* ── Bottom nav (model selection ↔ conversation) ─────────────────── */}
        {!isDownloading && currentPage !== 'ragSetup' && (
          <View style={styles.tabBar}>
            <TouchableOpacity
              style={[
                styles.tabBtn,
                currentPage === 'modelSelection' && styles.tabBtnActive,
              ]}
              onPress={() => setCurrentPage('modelSelection')}>
              <Text
                style={[
                  styles.tabBtnText,
                  currentPage === 'modelSelection' && styles.tabBtnTextActive,
                ]}>
                Models
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.tabBtn,
                currentPage === 'conversation' && styles.tabBtnActive,
                !llmContext && styles.tabBtnDisabled,
              ]}
              disabled={!llmContext}
              onPress={() => setCurrentPage('conversation')}>
              <Text
                style={[
                  styles.tabBtnText,
                  currentPage === 'conversation' && styles.tabBtnTextActive,
                  !llmContext && styles.tabBtnTextDisabled,
                ]}>
                Chat {llmContext ? '●' : '○'}
              </Text>
            </TouchableOpacity>
          </View>
        )}
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// STYLES
// ─────────────────────────────────────────────────────────────────────────────

const BLUE = '#3B82F6';
const BLUE_LIGHT = '#93C5FD';
const BLUE_DARK = '#1D4ED8';
const SURFACE = '#F8FAFC';
const CARD_BG = '#FFFFFF';
const BORDER = '#E2E8F0';
const TEXT_MAIN = '#1E293B';
const TEXT_SUB = '#64748B';
const TEXT_MUTED = '#94A3B8';
const GREEN = '#22C55E';
const RED = '#EF4444';

const styles = StyleSheet.create({
  root: {flex: 1, backgroundColor: SURFACE},
  flex: {flex: 1},

  // ── Header ────────────────────────────────────────────────────────────────
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: CARD_BG,
    borderBottomWidth: 1,
    borderBottomColor: BORDER,
    elevation: 2,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: TEXT_MAIN,
    letterSpacing: 0.3,
  },
  headerRight: {flexDirection: 'row', gap: 8},
  headerBtn: {
    backgroundColor: BLUE_LIGHT,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  headerBtnText: {fontSize: 13, fontWeight: '600', color: BLUE_DARK},

  // ── Status bar ────────────────────────────────────────────────────────────
  statusBar: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 6,
  },
  statusReady: {backgroundColor: '#16A34A'},
  statusLoading: {backgroundColor: '#F59E0B'},
  statusText: {fontSize: 12, color: '#FFFFFF', fontWeight: '500'},

  // ── Scroll pages ──────────────────────────────────────────────────────────
  scrollContent: {padding: 16, paddingBottom: 32},
  pageTitle: {
    fontSize: 22,
    fontWeight: '700',
    color: TEXT_MAIN,
    marginBottom: 16,
    marginTop: 4,
  },
  card: {
    backgroundColor: CARD_BG,
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: BORDER,
    elevation: 1,
  },
  sectionLabel: {
    fontSize: 13,
    fontWeight: '700',
    color: TEXT_SUB,
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    marginBottom: 12,
  },

  // ── Option buttons (model / GGUF selection) ───────────────────────────────
  optionBtn: {
    borderWidth: 1.5,
    borderColor: BORDER,
    borderRadius: 12,
    paddingVertical: 12,
    paddingHorizontal: 16,
    marginBottom: 8,
    backgroundColor: SURFACE,
  },
  optionBtnSelected: {borderColor: BLUE, backgroundColor: '#EFF6FF'},
  optionBtnText: {fontSize: 15, color: TEXT_MAIN, fontWeight: '500'},
  optionBtnTextSmall: {fontSize: 12, color: TEXT_MAIN, fontWeight: '400'},
  optionBtnTextSelected: {color: BLUE, fontWeight: '700'},

  // ── Download page ─────────────────────────────────────────────────────────
  centeredCard: {
    flex: 1,
    justifyContent: 'center',
    padding: 32,
    margin: 24,
    backgroundColor: CARD_BG,
    borderRadius: 20,
    elevation: 2,
    borderWidth: 1,
    borderColor: BORDER,
  },
  smallLabel: {fontSize: 12, color: BLUE_LIGHT, marginBottom: 16, fontWeight: '500'},
  progressPct: {
    textAlign: 'center',
    marginTop: 8,
    fontSize: 18,
    fontWeight: '700',
    color: BLUE,
  },

  // ── RAG Setup ─────────────────────────────────────────────────────────────
  toggleBtn: {
    paddingVertical: 12,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 4,
  },
  toggleBtnOn: {backgroundColor: GREEN},
  toggleBtnOff: {backgroundColor: BORDER},
  toggleBtnDisabled: {opacity: 0.5},
  toggleBtnText: {fontSize: 15, fontWeight: '700', color: '#FFFFFF'},
  hintText: {fontSize: 12, color: TEXT_MUTED, marginTop: 8},

  docInput: {
    borderWidth: 1,
    borderColor: BORDER,
    borderRadius: 12,
    padding: 12,
    fontSize: 14,
    color: TEXT_MAIN,
    minHeight: 160,
    marginBottom: 12,
    backgroundColor: SURFACE,
  },
  ingestProgress: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingVertical: 12,
  },
  ingestProgressText: {fontSize: 13, color: TEXT_SUB},

  primaryBtn: {
    backgroundColor: BLUE,
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
  },
  primaryBtnDisabled: {opacity: 0.4},
  primaryBtnText: {fontSize: 15, fontWeight: '700', color: '#FFFFFF'},

  dangerBtn: {
    borderWidth: 1.5,
    borderColor: RED,
    paddingVertical: 12,
    borderRadius: 12,
    alignItems: 'center',
  },
  dangerBtnText: {fontSize: 14, color: RED, fontWeight: '600'},

  // ── Chat ──────────────────────────────────────────────────────────────────
  chatScroll: {flex: 1},
  chatContent: {padding: 16, paddingBottom: 8},

  ragBadge: {
    backgroundColor: '#EFF6FF',
    borderRadius: 20,
    paddingHorizontal: 14,
    paddingVertical: 6,
    alignSelf: 'center',
    marginBottom: 12,
    borderWidth: 1,
    borderColor: BLUE_LIGHT,
  },
  ragBadgeText: {fontSize: 12, color: BLUE, fontWeight: '600'},

  emptyState: {
    flex: 1,
    alignItems: 'center',
    paddingTop: 60,
    paddingHorizontal: 32,
  },
  emptyStateText: {
    fontSize: 15,
    color: TEXT_MUTED,
    textAlign: 'center',
    lineHeight: 22,
  },

  bubbleWrapper: {marginBottom: 12},
  bubbleWrapperUser: {alignItems: 'flex-end'},
  bubbleWrapperAssistant: {alignItems: 'flex-start'},
  roleLabel: {fontSize: 11, color: TEXT_MUTED, marginBottom: 4, marginLeft: 4},

  bubble: {
    maxWidth: '82%',
    padding: 12,
    borderRadius: 16,
  },
  bubbleUser: {
    backgroundColor: BLUE,
    borderBottomRightRadius: 4,
  },
  bubbleAssistant: {
    backgroundColor: CARD_BG,
    borderWidth: 1,
    borderColor: BORDER,
    borderBottomLeftRadius: 4,
    elevation: 1,
  },
  bubbleText: {fontSize: 15, color: TEXT_MAIN, lineHeight: 22},
  bubbleTextUser: {color: '#FFFFFF'},
  retrievedPanel: {
    width: '92%',
    marginTop: 8,
    padding: 12,
    borderRadius: 12,
    backgroundColor: '#F8FAFC',
    borderWidth: 1,
    borderColor: BORDER,
  },
  retrievedTitle: {
    fontSize: 12,
    fontWeight: '700',
    color: TEXT_SUB,
    marginBottom: 8,
    textTransform: 'uppercase',
    letterSpacing: 0.6,
  },
  chunkCard: {
    paddingTop: 8,
    marginTop: 8,
    borderTopWidth: 1,
    borderTopColor: BORDER,
  },
  chunkMeta: {
    fontSize: 11,
    fontWeight: '700',
    color: BLUE_DARK,
    marginBottom: 4,
  },
  chunkText: {fontSize: 12, color: TEXT_MAIN, lineHeight: 18},
  chunkSource: {fontSize: 10, color: TEXT_MUTED, marginTop: 6},

  // ── Input bar ─────────────────────────────────────────────────────────────
  inputBar: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: 8,
    padding: 12,
    backgroundColor: CARD_BG,
    borderTopWidth: 1,
    borderTopColor: BORDER,
  },
  ragToggleChip: {
    paddingHorizontal: 10,
    paddingVertical: 8,
    borderRadius: 20,
    alignSelf: 'flex-end',
  },
  ragToggleChipOn: {backgroundColor: GREEN},
  ragToggleChipOff: {backgroundColor: BORDER},
  ragToggleChipText: {fontSize: 12, fontWeight: '700', color: '#FFFFFF'},

  textInput: {
    flex: 1,
    borderWidth: 1,
    borderColor: BORDER,
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 10,
    fontSize: 15,
    color: TEXT_MAIN,
    maxHeight: 120,
    backgroundColor: SURFACE,
  },
  sendBtn: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: BLUE,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sendBtnDisabled: {backgroundColor: BORDER},
  sendBtnText: {fontSize: 20, color: '#FFFFFF', fontWeight: '700'},

  // ── Tab bar ───────────────────────────────────────────────────────────────
  tabBar: {
    flexDirection: 'row',
    borderTopWidth: 1,
    borderTopColor: BORDER,
    backgroundColor: CARD_BG,
  },
  tabBtn: {
    flex: 1,
    paddingVertical: 14,
    alignItems: 'center',
  },
  tabBtnActive: {borderTopWidth: 2, borderTopColor: BLUE},
  tabBtnDisabled: {opacity: 0.4},
  tabBtnText: {fontSize: 14, color: TEXT_MUTED, fontWeight: '500'},
  tabBtnTextActive: {color: BLUE, fontWeight: '700'},
  tabBtnTextDisabled: {color: TEXT_MUTED},
});

export default App;
