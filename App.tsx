import React, {useState, useEffect, useRef} from 'react';
import {
  StyleSheet,
  View,
  Alert,
  ScrollView,
  Text,
  TouchableOpacity,
  ActivityIndicator,
  TextInput,
  Platform,
  KeyboardAvoidingView,
} from 'react-native';
import {SafeAreaProvider, SafeAreaView} from 'react-native-safe-area-context';

import RNFS from 'react-native-fs';
import {initLlama, releaseAllLlama} from 'llama.rn';

import {downloadModel, isModelDownloaded, GEMMA_MODEL} from './src/api/model';
import ProgressBar from './src/components/ProgressBar';

import {loadEmbeddingModel} from './src/api/embeddingModel';
import {initVectorDB, type RetrievedChunk} from './src/api/vectorDb';
import {buildRAGPrompt, type RAGDecision, type LLMCompletionFn} from './src/api/rag';

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

type ChatTurn = {
  userDisplay: string;
  userPrompt: string;
  assistantReply: string;
  ragChunks: RetrievedChunk[];
  ragDecision: RAGDecision | null;
};

type Page = 'home' | 'conversation' | 'ragSetup';

// ─────────────────────────────────────────────────────────────────────────────
// System prompt — Gemma uses <start_of_turn> / <end_of_turn> tags.
// We hardcode Telugu as the output language.
// ─────────────────────────────────────────────────────────────────────────────

const SYSTEM_CONTENT =
  'మీరు తెలుగు వ్యవసాయ AI సహాయకుడు. ' +
  'మీకు వ్యవసాయం, పంటలు, నేల, ఎరువులు, తెగుళ్ళు, నీటిపారుదల మరియు ' +
  'సంబంధిత అంశాలపై నిపుణత ఉంది. ' +
  'ముఖ్యమైనది: ఎల్లప్పుడూ తెలుగు భాషలో మాత్రమే సమాధానం ఇవ్వాలి. ' +
  'ఇంగ్లీష్ లేదా ఇతర భాషలు ఉపయోగించకూడదు. ' +
  'సందర్భం అందించబడినప్పుడు దానిని ఉపయోగించి సమాధానం ఇవ్వండి. ' +
  'సందర్భం లేనప్పుడు మీ స్వంత జ్ఞానం నుండి సమాధానం ఇవ్వండి. ' +
  'సమాధానాలు సంక్షిప్తంగా (200 పదాల కంటే తక్కువ) ఉండాలి.';

// Classifier system prompt (English only — for YES/NO answer)
const CLASSIFIER_SYSTEM =
  'You are a strict binary topic classifier. ' +
  'Reply with exactly one word: YES or NO. Nothing else.';

const STOP_WORDS = [
  '<end_of_turn>',
  '<eos>',
  '<|end_of_turn|>',
  '</s>',
  '<|endoftext|>',
];

// For the fast classifier we use fewer tokens
const CLASSIFIER_STOP_WORDS = [...STOP_WORDS, '\n', '.', ','];

const MAX_TURNS_IN_CONTEXT = 2;

const DECISION_META: Record<
  RAGDecision,
  {label: string; color: string; bg: string}
> = {
  rag_good:      {label: '📄 RAG — మంచి సరిపోలిక',       color: '#166534', bg: '#DCFCE7'},
  rag_weak:      {label: '⚠️ RAG — బలహీన సరిపోలిక',       color: '#92400E', bg: '#FEF3C7'},
  rag_no_chunks: {label: '🔍 వ్యవసాయ ప్రశ్న — chunks లేవు', color: '#1E3A5F', bg: '#DBEAFE'},
  skip_not_agri: {label: '💬 సాధారణ — LLM మాత్రమే',        color: '#4B1D96', bg: '#EDE9FE'},
};

// ─────────────────────────────────────────────────────────────────────────────

function App(): React.JSX.Element {
  const [llmContext, setLlmContext]               = useState<any>(null);
  const [downloadProgress, setDownloadProgress]   = useState(0);
  const [isDownloading, setIsDownloading]         = useState(false);
  const [modelReady, setModelReady]               = useState(false);

  const [turns, setTurns]                         = useState<ChatTurn[]>([]);
  const [streamingText, setStreamingText]         = useState('');
  const [isGenerating, setIsGenerating]           = useState(false);
  const [userInput, setUserInput]                 = useState('');

  const [embeddingReady, setEmbeddingReady]       = useState(false);
  const [embeddingLoading, setEmbeddingLoading]   = useState(false);
  const [ragEnabled, setRagEnabled]               = useState(false);
  const [chunksLoaded, setChunksLoaded]           = useState(false);
  const [knowledgeChunkCount, setKnowledgeChunkCount] = useState(0);
  const [isIngesting, setIsIngesting]             = useState(false);

  // Shows "classifying…" spinner while LLM decides agri vs non-agri
  const [isClassifying, setIsClassifying]         = useState(false);

  const [currentPage, setCurrentPage]             = useState<Page>('home');

  const scrollRef = useRef<ScrollView>(null);

  // ── Mount ──────────────────────────────────────────────────────────────────
  useEffect(() => {
    const setup = async () => {
      const already = await isModelDownloaded();
      if (already) await loadLLM(false);

      try {
        const count = await initVectorDB();
        setKnowledgeChunkCount(count);
        setChunksLoaded(count > 0);
        setRagEnabled(count > 0);
      } catch (e) {
        console.warn('VectorDB init failed:', e);
      }

      setEmbeddingLoading(true);
      try {
        await loadEmbeddingModel();
        setEmbeddingReady(true);
      } catch (e) {
        console.warn('Embedding model failed:', e);
        Alert.alert(
          'Embedding Warning',
          'Could not load embedding model. RAG will be disabled.',
        );
      } finally {
        setEmbeddingLoading(false);
      }
    };
    setup();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    setTimeout(() => scrollRef.current?.scrollToEnd({animated: true}), 100);
  }, [turns, streamingText]);

  // ── Load LLM ───────────────────────────────────────────────────────────────
  const loadLLM = async (showAlert = true): Promise<boolean> => {
    try {
      const destPath = `${RNFS.DocumentDirectoryPath}/${GEMMA_MODEL.name}`;
      const exists = await RNFS.exists(destPath);
      if (!exists) {
        Alert.alert('Error', 'Model file not found. Please download it first.');
        return false;
      }

      const stat = await RNFS.stat(destPath);
      if (stat.size < 10_000_000) {
        await RNFS.unlink(destPath);
        Alert.alert(
          'Corrupt Model File',
          'The saved model file is too small and is likely corrupt.\n\n' +
            'The file has been deleted. Please download again.',
        );
        setModelReady(false);
        return false;
      }

      if (llmContext) {
        await releaseAllLlama();
        setLlmContext(null);
        setTurns([]);
      }

      const ctx = await initLlama({
        model: destPath,
        use_mlock: true,
        n_ctx: 2048,
        n_gpu_layers: 0,
        n_batch: 512,
        n_threads: 4,
      });

      setLlmContext(ctx);
      setModelReady(true);
      if (showAlert) Alert.alert('సిద్ధం!', `${GEMMA_MODEL.displayName} లోడ్ అయింది!`);
      return true;
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      console.error('LLM load error:', msg);
      Alert.alert('Load Error', `Could not load model.\n\n${msg}`);
      return false;
    }
  };

  // ── Download ───────────────────────────────────────────────────────────────
  const handleDownload = async () => {
    setIsDownloading(true);
    setDownloadProgress(0);
    try {
      await downloadModel(GEMMA_MODEL.name, GEMMA_MODEL.url, p =>
        setDownloadProgress(p),
      );
      const loaded = await loadLLM(true);
      if (loaded) setCurrentPage('conversation');
    } catch (error) {
      Alert.alert(
        'Download Error',
        error instanceof Error ? error.message : 'Unknown error',
      );
    } finally {
      setIsDownloading(false);
    }
  };

  // ── Delete model ───────────────────────────────────────────────────────────
  const handleDeleteModel = async () => {
    Alert.alert('Delete Model', 'Delete the downloaded model file?', [
      {text: 'Cancel', style: 'cancel'},
      {
        text: 'Delete',
        style: 'destructive',
        onPress: async () => {
          try {
            if (llmContext) {
              await releaseAllLlama();
              setLlmContext(null);
              setModelReady(false);
              setTurns([]);
            }
            const destPath = `${RNFS.DocumentDirectoryPath}/${GEMMA_MODEL.name}`;
            if (await RNFS.exists(destPath)) await RNFS.unlink(destPath);
            Alert.alert('Deleted', 'Model removed. Download again to use.');
          } catch (e) {
            Alert.alert('Error', e instanceof Error ? e.message : 'Unknown');
          }
        },
      },
    ]);
  };

  // ── LLMCompletionFn ────────────────────────────────────────────────────────
  // This thin wrapper is passed into buildRAGPrompt so rag.ts can call the
  // LLM for classification without depending on llama.rn directly.
  const makeLLMCompleteFn = (ctx: any): LLMCompletionFn =>
    async (messages) => {
      let reply = '';
      await ctx.completion(
        {
          messages,
          n_predict: 8,           // classifier only needs YES / NO (a bit of room)
          stop: CLASSIFIER_STOP_WORDS,
          temperature: 0.0,       // deterministic
          top_p: 1.0,
          repeat_penalty: 1.0,
        },
        (data: {token: string}) => {
          reply += data.token;
        },
      );
      return reply.trim();
    };

  // ── Build sliding-window messages (Gemma chat format) ─────────────────────
  const buildLLMMessages = (
    allTurns: ChatTurn[],
    pendingPrompt: string,
  ) => {
    const windowTurns = allTurns.slice(-MAX_TURNS_IN_CONTEXT);
    const messages: {role: 'system' | 'user' | 'assistant'; content: string}[] =
      [{role: 'system', content: SYSTEM_CONTENT}];
    for (const t of windowTurns) {
      messages.push({role: 'user',      content: t.userPrompt});
      messages.push({role: 'assistant', content: t.assistantReply});
    }
    messages.push({role: 'user', content: pendingPrompt});
    return messages;
  };

  // ── Send message ───────────────────────────────────────────────────────────
  const handleSendMessage = async () => {
    if (!llmContext) {
      Alert.alert('No Model', 'Please download the model first.');
      return;
    }
    if (!userInput.trim()) return;

    const userText = userInput.trim();
    setUserInput('');
    setIsGenerating(true);
    setStreamingText('');

    try {
      let promptContent = userText;
      let retrievedChunks: RetrievedChunk[] = [];
      let decision: RAGDecision | null = null;

      if (ragEnabled && chunksLoaded && embeddingReady) {
        try {
          // Show "classifying" indicator while LLM decides agri vs non-agri
          setIsClassifying(true);
          const llmCompleteFn = makeLLMCompleteFn(llmContext);
          const ragResult = await buildRAGPrompt(userText, llmCompleteFn);
          setIsClassifying(false);

          promptContent   = ragResult.prompt;
          retrievedChunks = ragResult.chunks;
          decision        = ragResult.decision;
        } catch (ragErr) {
          setIsClassifying(false);
          console.warn('RAG failed, falling back to plain question:', ragErr);
        }
      }

      const llmMessages = buildLLMMessages(turns, promptContent);

      let fullReply = '';
      const result = await llmContext.completion(
        {
          messages: llmMessages,
          n_predict: 400,
          stop: STOP_WORDS,
          temperature: 0.7,
          top_p: 0.9,
          repeat_penalty: 1.1,
        },
        (data: {token: string}) => {
          fullReply += data.token;
          setStreamingText(fullReply);
        },
      );

      const finalReply = ((result?.text ?? '') || fullReply).trim();
      if (!finalReply) throw new Error('Empty response from model.');

      setTurns(prev => [
        ...prev,
        {
          userDisplay:    userText,
          userPrompt:     promptContent,
          assistantReply: finalReply,
          ragChunks:      retrievedChunks,
          ragDecision:    decision,
        },
      ]);
      setStreamingText('');
    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Unknown error.';
      Alert.alert('Inference Error', msg);
      setStreamingText('');
      setIsClassifying(false);
    } finally {
      setIsGenerating(false);
    }
  };

  // ── Reload knowledge DB ────────────────────────────────────────────────────
  const handleReloadDB = async () => {
    if (!embeddingReady) {
      Alert.alert('Not Ready', 'Embedding model is still loading.');
      return;
    }
    setIsIngesting(true);
    try {
      const count = await initVectorDB();
      setKnowledgeChunkCount(count);
      setChunksLoaded(count > 0);
      setRagEnabled(count > 0);
      Alert.alert('Done', `Knowledge DB loaded with ${count} chunks.`);
    } catch (e) {
      Alert.alert('DB Error', e instanceof Error ? e.message : 'Unknown');
    } finally {
      setIsIngesting(false);
    }
  };

  const turnsInWindow = Math.min(turns.length, MAX_TURNS_IN_CONTEXT);
  const turnsDropped  = Math.max(0, turns.length - MAX_TURNS_IN_CONTEXT);

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.root}>
        <KeyboardAvoidingView
          style={styles.flex}
          behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
          keyboardVerticalOffset={0}>

          {/* ── Header ──────────────────────────────────────────────────── */}
          <View style={styles.header}>
            <Text style={styles.headerTitle}>🌾 AgriRAG తెలుగు</Text>
            <View style={styles.headerRight}>
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

          {/* ── Embedding status bar ─────────────────────────────────────── */}
          <View style={[
            styles.statusBar,
            embeddingReady ? styles.statusReady : styles.statusLoading,
          ]}>
            {embeddingLoading ? (
              <>
                <ActivityIndicator size="small" color="#fff" style={{marginRight: 8}} />
                <Text style={styles.statusText}>Embedding model లోడ్ అవుతోంది…</Text>
              </>
            ) : embeddingReady ? (
              <Text style={styles.statusText}>
                ✓ Embedding సిద్ధం
                {chunksLoaded ? `  |  ✓ ${knowledgeChunkCount} chunks` : '  |  DB లేదు'}
                {chunksLoaded ? (ragEnabled ? '  |  RAG ON' : '  |  RAG OFF') : ''}
              </Text>
            ) : (
              <Text style={styles.statusText}>⚠ Embedding విఫలమైంది</Text>
            )}
          </View>

          {/* ══════════════════ PAGE: HOME ══════════════════ */}
          {currentPage === 'home' && !isDownloading && (
            <ScrollView contentContainerStyle={styles.scrollContent}>
              <Text style={styles.pageTitle}>ప్రారంభించండి</Text>

              <View style={styles.card}>
                <Text style={styles.sectionLabel}>Model</Text>
                <Text style={styles.modelName}>{GEMMA_MODEL.displayName}</Text>
                <Text style={styles.modelMeta}>
                  Quantization: Q4_K_S  ·  {GEMMA_MODEL.sizeHint}
                </Text>
                <Text style={[styles.hintText, {marginTop: 6}]}>
                  తెలుగు భాషలో వ్యవసాయ సహాయం అందించే Gemma 3 1B మోడల్.
                </Text>

                {modelReady ? (
                  <View style={{gap: 10, marginTop: 16}}>
                    <TouchableOpacity
                      style={styles.primaryBtn}
                      onPress={() => setCurrentPage('conversation')}>
                      <Text style={styles.primaryBtnText}>Chat తెరవండి →</Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                      style={styles.dangerBtn}
                      onPress={handleDeleteModel}>
                      <Text style={styles.dangerBtnText}>Model File తొలగించండి</Text>
                    </TouchableOpacity>
                  </View>
                ) : (
                  <TouchableOpacity
                    style={[styles.primaryBtn, {marginTop: 16}]}
                    onPress={handleDownload}>
                    <Text style={styles.primaryBtnText}>
                      ⬇ Download &amp; Load Model
                    </Text>
                  </TouchableOpacity>
                )}
              </View>

              <View style={styles.card}>
                <Text style={styles.sectionLabel}>Smart RAG Logic (LLM నిర్ణయిస్తుంది)</Text>
                <Text style={styles.hintText}>
                  {'🤖 మీ ప్రశ్న → LLM వ్యవసాయమా కాదా అని నిర్ణయిస్తుంది\n'}
                  {'🌾 వ్యవసాయ + మంచి chunks → RAG సమాధానం\n'}
                  {'⚠️  వ్యవసాయ + బలహీన chunks → RAG + LLM కలిపి\n'}
                  {'🔍 వ్యవసాయ + chunks లేవు → LLM స్వయంగా సమాధానం\n'}
                  {'💬 వ్యవసాయం కాదు → LLM మాత్రమే (RAG లేదు)'}
                </Text>
              </View>

              <View style={styles.card}>
                <Text style={styles.sectionLabel}>Token Budget / Turn</Text>
                <Text style={styles.hintText}>
                  {'Question      →  128 tokens\n'}
                  {'3 chunks×150  →  ~450 tokens\n'}
                  {'LLM reply     →  ~300 tokens\n'}
                  {'Classifier    →  ~50 tokens\n'}
                  {'─────────────────────────────\n'}
                  {`Per turn ≈ 1 000 tokens\n`}
                  {`${MAX_TURNS_IN_CONTEXT} turns kept in context ≈ 2 000 tokens (n_ctx=2048)`}
                </Text>
              </View>
            </ScrollView>
          )}

          {/* ══════════════════ PAGE: DOWNLOADING ══════════════════ */}
          {isDownloading && (
            <View style={styles.centeredCard}>
              <Text style={styles.sectionLabel}>డౌన్లోడ్ అవుతోంది</Text>
              <Text style={styles.modelName}>{GEMMA_MODEL.displayName}</Text>
              <Text style={styles.modelMeta}>{GEMMA_MODEL.sizeHint}</Text>
              <ProgressBar progress={downloadProgress / 100} />
              <Text style={styles.progressPct}>{downloadProgress}%</Text>
              <Text style={[styles.hintText, {textAlign: 'center', marginTop: 12}]}>
                100% వద్ద ఆగిపోతే, model URL మార్చాల్సి ఉండవచ్చు.
              </Text>
            </View>
          )}

          {/* ══════════════════ PAGE: RAG SETUP ══════════════════ */}
          {currentPage === 'ragSetup' && (
            <ScrollView contentContainerStyle={styles.scrollContent}>
              <Text style={styles.pageTitle}>Document Setup</Text>

              <View style={styles.card}>
                <Text style={styles.sectionLabel}>RAG Mode</Text>
                <TouchableOpacity
                  style={[
                    styles.toggleBtn,
                    ragEnabled ? styles.toggleBtnOn : styles.toggleBtnOff,
                    (!chunksLoaded || !embeddingReady) && styles.toggleBtnDisabled,
                  ]}
                  disabled={!chunksLoaded || !embeddingReady}
                  onPress={() => setRagEnabled(p => !p)}>
                  <Text style={styles.toggleBtnText}>
                    {ragEnabled ? '✓ RAG Enabled' : 'RAG Disabled'}
                  </Text>
                </TouchableOpacity>
                {!chunksLoaded && (
                  <Text style={styles.hintText}>
                    Knowledge DB లోడ్ చేయండి.
                  </Text>
                )}
              </View>

              <View style={styles.card}>
                <Text style={styles.sectionLabel}>Knowledge Database</Text>
                <Text style={styles.hintText}>
                  {'Bundled: knowledge.db (Python లో నిర్మించబడింది)\n'}
                  {knowledgeChunkCount
                    ? `Chunks: ${knowledgeChunkCount}`
                    : 'ఇంకా chunks లోడ్ కాలేదు.'}
                </Text>
                {isIngesting ? (
                  <View style={styles.ingestProgress}>
                    <ActivityIndicator size="small" color={BLUE} />
                    <Text style={styles.ingestProgressText}>లోడ్ అవుతోంది…</Text>
                  </View>
                ) : (
                  <TouchableOpacity
                    style={[
                      styles.primaryBtn,
                      {marginTop: 12},
                      !embeddingReady && styles.primaryBtnDisabled,
                    ]}
                    disabled={!embeddingReady}
                    onPress={handleReloadDB}>
                    <Text style={styles.primaryBtnText}>Knowledge DB రీలోడ్</Text>
                  </TouchableOpacity>
                )}
              </View>
            </ScrollView>
          )}

          {/* ══════════════════ PAGE: CONVERSATION ══════════════════ */}
          {currentPage === 'conversation' && !isDownloading && (
            <>
              <ScrollView
                ref={scrollRef}
                style={styles.chatScroll}
                contentContainerStyle={styles.chatContent}>

                {ragEnabled && chunksLoaded && (
                  <View style={styles.ragBadge}>
                    <Text style={styles.ragBadgeText}>
                      🌾 Smart RAG active · LLM నిర్ణయం
                    </Text>
                  </View>
                )}

                {turns.length > 0 && (
                  <View style={styles.windowBanner}>
                    <Text style={styles.windowBannerText}>
                      Context: {turnsInWindow}/{MAX_TURNS_IN_CONTEXT} turns
                      {turnsDropped > 0
                        ? `  ·  ${turnsDropped} turn${turnsDropped > 1 ? 's' : ''} తీసివేయబడ్డాయి`
                        : ''}
                    </Text>
                  </View>
                )}

                {!llmContext && (
                  <View style={styles.emptyState}>
                    <Text style={styles.emptyStateText}>
                      ముందుగా Home లో model డౌన్లోడ్ చేయండి.
                    </Text>
                  </View>
                )}

                {turns.map((turn, idx) => {
                  const isOutside = idx < turns.length - MAX_TURNS_IN_CONTEXT;
                  const dm = turn.ragDecision ? DECISION_META[turn.ragDecision] : null;

                  return (
                    <View key={idx}>
                      {isOutside && (
                        <Text style={styles.outsideWindowLabel}>
                          ↑ LLM context బయట
                        </Text>
                      )}

                      {/* User bubble */}
                      <View style={[
                        styles.bubbleWrapper,
                        styles.bubbleWrapperUser,
                        isOutside && styles.fadedOpacity,
                      ]}>
                        <View style={[styles.bubble, styles.bubbleUser]}>
                          <Text style={[styles.bubbleText, styles.bubbleTextUser]}>
                            {turn.userDisplay}
                          </Text>
                        </View>

                        {dm && (
                          <View style={[styles.decisionBadge, {backgroundColor: dm.bg}]}>
                            <Text style={[styles.decisionBadgeText, {color: dm.color}]}>
                              {dm.label}
                            </Text>
                          </View>
                        )}

                        {turn.ragChunks.length > 0 && (
                          <View style={styles.retrievedPanel}>
                            <Text style={styles.retrievedTitle}>
                              {turn.ragChunks.length} chunk
                              {turn.ragChunks.length > 1 ? 's' : ''} retrieved
                              {' '}(≤150 words each)
                            </Text>
                            {turn.ragChunks.map((chunk, ci) => (
                              <View key={`${chunk.id}-${ci}`} style={styles.chunkCard}>
                                <Text style={styles.chunkMeta}>
                                  #{ci + 1} · id:{chunk.id}
                                  {chunk.page != null ? ` · p.${chunk.page}` : ''}
                                  {Number.isFinite(Number(chunk.distance))
                                    ? ` · dist:${Number(chunk.distance).toFixed(3)}`
                                    : ''}
                                </Text>
                                <Text style={styles.chunkText}>{chunk.text}</Text>
                                {chunk.source ? (
                                  <Text style={styles.chunkSource}>{chunk.source}</Text>
                                ) : null}
                              </View>
                            ))}
                          </View>
                        )}
                      </View>

                      {/* Assistant bubble */}
                      <View style={[
                        styles.bubbleWrapper,
                        styles.bubbleWrapperAssistant,
                        isOutside && styles.fadedOpacity,
                      ]}>
                        <Text style={styles.roleLabel}>🌾 Gemma తెలుగు</Text>
                        <View style={[styles.bubble, styles.bubbleAssistant]}>
                          <Text style={styles.bubbleText}>{turn.assistantReply}</Text>
                        </View>
                      </View>
                    </View>
                  );
                })}

                {/* Classifying spinner (shown before streaming starts) */}
                {isClassifying && !streamingText && (
                  <View style={[styles.bubbleWrapper, styles.bubbleWrapperAssistant]}>
                    <Text style={styles.roleLabel}>🤖 వర్గీకరిస్తోంది…</Text>
                    <View style={[styles.bubble, styles.bubbleAssistant, styles.classifyingBubble]}>
                      <ActivityIndicator size="small" color={BLUE} />
                      <Text style={styles.classifyingText}>
                        ప్రశ్న విశ్లేషిస్తోంది…
                      </Text>
                    </View>
                  </View>
                )}

                {/* Streaming bubble */}
                {isGenerating && !isClassifying && (
                  <View style={[styles.bubbleWrapper, styles.bubbleWrapperAssistant]}>
                    <Text style={styles.roleLabel}>🌾 Gemma తెలుగు</Text>
                    <View style={[styles.bubble, styles.bubbleAssistant]}>
                      {streamingText ? (
                        <Text style={styles.bubbleText}>{streamingText}</Text>
                      ) : (
                        <ActivityIndicator size="small" color={BLUE} />
                      )}
                    </View>
                  </View>
                )}
              </ScrollView>

              {/* Input bar */}
              <View style={styles.inputBar}>
                {embeddingReady && chunksLoaded && (
                  <TouchableOpacity
                    style={[
                      styles.ragToggleChip,
                      ragEnabled ? styles.ragToggleChipOn : styles.ragToggleChipOff,
                    ]}
                    onPress={() => setRagEnabled(p => !p)}>
                    <Text style={styles.ragToggleChipText}>
                      {ragEnabled ? '🌾' : '💬'}
                    </Text>
                  </TouchableOpacity>
                )}
                <TextInput
                  style={styles.textInput}
                  placeholder={
                    llmContext
                      ? ragEnabled
                        ? 'పంటలు, నేల, తెగుళ్ళ గురించి అడగండి…'
                        : 'సందేశం టైప్ చేయండి…'
                      : 'ముందుగా model లోడ్ చేయండి…'
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

          {/* Tab bar */}
          {!isDownloading && currentPage !== 'ragSetup' && (
            <View style={styles.tabBar}>
              <TouchableOpacity
                style={[styles.tabBtn, currentPage === 'home' && styles.tabBtnActive]}
                onPress={() => setCurrentPage('home')}>
                <Text style={[
                  styles.tabBtnText,
                  currentPage === 'home' && styles.tabBtnTextActive,
                ]}>Home</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[
                  styles.tabBtn,
                  currentPage === 'conversation' && styles.tabBtnActive,
                  !llmContext && styles.tabBtnDisabled,
                ]}
                disabled={!llmContext}
                onPress={() => setCurrentPage('conversation')}>
                <Text style={[
                  styles.tabBtnText,
                  currentPage === 'conversation' && styles.tabBtnTextActive,
                  !llmContext && styles.tabBtnTextDisabled,
                ]}>Chat {llmContext ? '●' : '○'}</Text>
              </TouchableOpacity>
            </View>
          )}

        </KeyboardAvoidingView>
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// STYLES
// ─────────────────────────────────────────────────────────────────────────────

const BLUE       = '#3B82F6';
const BLUE_LIGHT = '#93C5FD';
const BLUE_DARK  = '#1D4ED8';
const SURFACE    = '#F8FAFC';
const CARD_BG    = '#FFFFFF';
const BORDER     = '#E2E8F0';
const TEXT_MAIN  = '#1E293B';
const TEXT_SUB   = '#64748B';
const TEXT_MUTED = '#94A3B8';
const GREEN      = '#22C55E';
const RED        = '#EF4444';

const styles = StyleSheet.create({
  root: {flex: 1, backgroundColor: SURFACE},
  flex: {flex: 1},

  header: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: 16, paddingVertical: 12,
    backgroundColor: CARD_BG, borderBottomWidth: 1, borderBottomColor: BORDER, elevation: 2,
  },
  headerTitle:   {fontSize: 20, fontWeight: '700', color: TEXT_MAIN},
  headerRight:   {flexDirection: 'row', gap: 8},
  headerBtn:     {backgroundColor: BLUE_LIGHT, paddingHorizontal: 12, paddingVertical: 6, borderRadius: 20},
  headerBtnText: {fontSize: 13, fontWeight: '600', color: BLUE_DARK},

  statusBar:    {flexDirection: 'row', alignItems: 'center', paddingHorizontal: 16, paddingVertical: 6},
  statusReady:  {backgroundColor: '#16A34A'},
  statusLoading:{backgroundColor: '#F59E0B'},
  statusText:   {fontSize: 12, color: '#FFFFFF', fontWeight: '500'},

  scrollContent: {padding: 16, paddingBottom: 32},
  pageTitle:     {fontSize: 22, fontWeight: '700', color: TEXT_MAIN, marginBottom: 16, marginTop: 4},
  card: {
    backgroundColor: CARD_BG, borderRadius: 16, padding: 16, marginBottom: 16,
    borderWidth: 1, borderColor: BORDER, elevation: 1,
  },
  sectionLabel: {
    fontSize: 13, fontWeight: '700', color: TEXT_SUB,
    textTransform: 'uppercase', letterSpacing: 0.8, marginBottom: 8,
  },
  modelName: {fontSize: 17, fontWeight: '700', color: TEXT_MAIN, marginBottom: 4},
  modelMeta: {fontSize: 12, color: TEXT_MUTED, marginBottom: 4},
  hintText:  {fontSize: 12, color: TEXT_MUTED, lineHeight: 20},

  centeredCard: {
    flex: 1, justifyContent: 'center', padding: 32, margin: 24,
    backgroundColor: CARD_BG, borderRadius: 20, elevation: 2, borderWidth: 1, borderColor: BORDER,
  },
  progressPct: {textAlign: 'center', marginTop: 8, fontSize: 18, fontWeight: '700', color: BLUE},

  toggleBtn:         {paddingVertical: 12, borderRadius: 12, alignItems: 'center', marginBottom: 4},
  toggleBtnOn:       {backgroundColor: GREEN},
  toggleBtnOff:      {backgroundColor: BORDER},
  toggleBtnDisabled: {opacity: 0.5},
  toggleBtnText:     {fontSize: 15, fontWeight: '700', color: '#FFFFFF'},

  ingestProgress:     {flexDirection: 'row', alignItems: 'center', gap: 10, paddingVertical: 12},
  ingestProgressText: {fontSize: 13, color: TEXT_SUB},

  primaryBtn:         {backgroundColor: BLUE, paddingVertical: 14, borderRadius: 12, alignItems: 'center'},
  primaryBtnDisabled: {opacity: 0.4},
  primaryBtnText:     {fontSize: 15, fontWeight: '700', color: '#FFFFFF'},
  dangerBtn:          {borderWidth: 1.5, borderColor: RED, paddingVertical: 12, borderRadius: 12, alignItems: 'center'},
  dangerBtnText:      {fontSize: 14, color: RED, fontWeight: '600'},

  chatScroll:  {flex: 1},
  chatContent: {padding: 16, paddingBottom: 8},

  ragBadge: {
    backgroundColor: '#ECFDF5', borderRadius: 20, paddingHorizontal: 14, paddingVertical: 6,
    alignSelf: 'center', marginBottom: 8, borderWidth: 1, borderColor: '#6EE7B7',
  },
  ragBadgeText: {fontSize: 12, color: '#065F46', fontWeight: '600'},

  windowBanner: {
    backgroundColor: '#FEF9C3', borderRadius: 10, paddingHorizontal: 12, paddingVertical: 5,
    alignSelf: 'center', marginBottom: 10, borderWidth: 1, borderColor: '#FDE047',
  },
  windowBannerText: {fontSize: 11, color: '#713F12', fontWeight: '500'},

  outsideWindowLabel: {
    fontSize: 10, color: TEXT_MUTED, textAlign: 'center',
    marginTop: 8, marginBottom: 2, fontStyle: 'italic',
  },
  fadedOpacity: {opacity: 0.45},

  emptyState:     {alignItems: 'center', paddingTop: 60, paddingHorizontal: 32},
  emptyStateText: {fontSize: 15, color: TEXT_MUTED, textAlign: 'center', lineHeight: 22},

  bubbleWrapper:          {marginBottom: 8},
  bubbleWrapperUser:      {alignItems: 'flex-end'},
  bubbleWrapperAssistant: {alignItems: 'flex-start'},
  roleLabel:              {fontSize: 11, color: TEXT_MUTED, marginBottom: 3, marginLeft: 4},

  bubble:          {maxWidth: '82%', padding: 12, borderRadius: 16},
  bubbleUser:      {backgroundColor: BLUE, borderBottomRightRadius: 4},
  bubbleAssistant: {
    backgroundColor: CARD_BG, borderWidth: 1, borderColor: BORDER,
    borderBottomLeftRadius: 4, elevation: 1,
  },
  bubbleText:     {fontSize: 15, color: TEXT_MAIN, lineHeight: 22},
  bubbleTextUser: {color: '#FFFFFF'},

  // Classifying indicator bubble
  classifyingBubble: {flexDirection: 'row', alignItems: 'center', gap: 8, paddingVertical: 10},
  classifyingText:   {fontSize: 13, color: TEXT_MUTED, fontStyle: 'italic'},

  decisionBadge: {
    alignSelf: 'flex-end', marginTop: 4, marginRight: 4,
    paddingHorizontal: 10, paddingVertical: 4, borderRadius: 20,
  },
  decisionBadgeText: {fontSize: 11, fontWeight: '600'},

  retrievedPanel: {
    width: '92%', marginTop: 6, padding: 10, borderRadius: 10,
    backgroundColor: '#F8FAFC', borderWidth: 1, borderColor: BORDER,
  },
  retrievedTitle: {
    fontSize: 11, fontWeight: '700', color: TEXT_SUB, marginBottom: 6,
    textTransform: 'uppercase', letterSpacing: 0.5,
  },
  chunkCard:   {paddingTop: 6, marginTop: 6, borderTopWidth: 1, borderTopColor: BORDER},
  chunkMeta:   {fontSize: 10, fontWeight: '700', color: BLUE_DARK, marginBottom: 3},
  chunkText:   {fontSize: 11, color: TEXT_MAIN, lineHeight: 17},
  chunkSource: {fontSize: 10, color: TEXT_MUTED, marginTop: 4},

  inputBar: {
    flexDirection: 'row', alignItems: 'flex-end', gap: 8, padding: 12,
    backgroundColor: CARD_BG, borderTopWidth: 1, borderTopColor: BORDER,
  },
  ragToggleChip:    {paddingHorizontal: 10, paddingVertical: 8, borderRadius: 20, alignSelf: 'flex-end'},
  ragToggleChipOn:  {backgroundColor: GREEN},
  ragToggleChipOff: {backgroundColor: BORDER},
  ragToggleChipText:{fontSize: 14, fontWeight: '700', color: '#FFFFFF'},

  textInput: {
    flex: 1, borderWidth: 1, borderColor: BORDER, borderRadius: 20,
    paddingHorizontal: 16, paddingVertical: 10, fontSize: 15, color: TEXT_MAIN,
    maxHeight: 120, backgroundColor: SURFACE,
  },
  sendBtn:         {width: 44, height: 44, borderRadius: 22, backgroundColor: BLUE, alignItems: 'center', justifyContent: 'center'},
  sendBtnDisabled: {backgroundColor: BORDER},
  sendBtnText:     {fontSize: 20, color: '#FFFFFF', fontWeight: '700'},

  tabBar:            {flexDirection: 'row', borderTopWidth: 1, borderTopColor: BORDER, backgroundColor: CARD_BG},
  tabBtn:            {flex: 1, paddingVertical: 14, alignItems: 'center'},
  tabBtnActive:      {borderTopWidth: 2, borderTopColor: BLUE},
  tabBtnDisabled:    {opacity: 0.4},
  tabBtnText:        {fontSize: 14, color: TEXT_MUTED, fontWeight: '500'},
  tabBtnTextActive:  {color: BLUE, fontWeight: '700'},
  tabBtnTextDisabled:{color: TEXT_MUTED},
});

export default App;