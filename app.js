"use strict";

const DEFAULT_LABELS = Array.from("ABCDEFGHIKLMNOPQRSTUVWXY");
const CONFIDENCE_THRESHOLDS = {
  N: 0.75,
  M: 0.72,
  T: 0.72,
  S: 0.7,
  default: 0.6,
};

const FEATURE_DIM = 84;
const FEATURE_BUFFER_SIZE = 3;
const RAW_PRED_BUFFER_SIZE = 5;
const LETTER_HOLD_THRESHOLD = 15;
const FAST_MODE_THRESHOLD = 8;
const REPEAT_DELAY_FRAMES = 45;
const SPELLING_COOLDOWN_DURATION = 45;

const VOCAB_STORAGE_KEY = "sign2sound_vocab_memory_v1";
const VOCAB_SEED_URL = "./vocab_memory_seed.json";
const VOCAB_N_ORDER = 5;
const SUGGESTION_TOP_K = 3;
const UNMIRROR_PREVIEW = true;
const CORE_DEFAULT_WORDS = [
  "HELLO",
  "YES",
  "NO",
  "GOOD",
  "BAD",
  "HELP",
  "STOP",
  "GO",
  "COME",
  "I",
  "YOU",
  "HE",
  "SHE",
  "IT",
  "WE",
  "THEY",
  "THE",
  "AND",
  "TO",
  "A",
];

const videoEl = document.getElementById("videoInput");
const canvasEl = document.getElementById("overlay");
const ctx = canvasEl.getContext("2d");
const cameraCardEl = document.querySelector(".camera-card");

const statusEl = document.getElementById("status");
const liveLetterEl = document.getElementById("liveLetter");
const liveConfidenceEl = document.getElementById("liveConfidence");
const holdProgressEl = document.getElementById("holdProgress");
const currentWordEl = document.getElementById("currentWord");
const transcriptEl = document.getElementById("transcript");
const suggestionsEl = document.getElementById("suggestions");
const modeBadgeEl = document.getElementById("modeBadge");
const gestureHintEl = document.getElementById("gestureHint");

const confirmWordBtn = document.getElementById("confirmWordBtn");
const acceptSuggestionBtn = document.getElementById("acceptSuggestionBtn");
const backspaceBtn = document.getElementById("backspaceBtn");
const undoWordBtn = document.getElementById("undoWordBtn");
const clearWordBtn = document.getElementById("clearWordBtn");
const speakWordBtn = document.getElementById("speakWordBtn");
const switchCameraBtn = document.getElementById("switchCameraBtn");

let labels = DEFAULT_LABELS.slice();
let model = null;

let featureBuffer = [];
let rawPredBuffer = [];

let letterBuffer = [];
let currentWord = "";
let transcriptWords = [];
let suggestions = [];

let lastDetectedLetter = null;
let letterHoldFrames = 0;
let spellingCooldownFrames = 0;

let vocabMemory = createEmptyVocabulary();
let gestureCtrl = null;
let hands = null;
let animationFrameId = null;
let isSendingFrame = false;
let stream = null;
let cameraDevices = [];
let activeDeviceId = null;
let activeFacingMode = "user";
let isSwitchingCamera = false;

function hasMultipleCameras() {
  return cameraDevices.length > 1;
}

function inferFacingModeFromLabel(labelRaw) {
  const label = String(labelRaw || "").toLowerCase();
  if (!label) {
    return null;
  }

  if (/(rear|back|environment|world)/.test(label)) {
    return "environment";
  }
  if (/(front|user|facetime|selfie)/.test(label)) {
    return "user";
  }
  return null;
}

function updateSwitchCameraButton() {
  if (!switchCameraBtn) {
    return;
  }
  const enabled = hasMultipleCameras();
  switchCameraBtn.disabled = !enabled;

  if (activeFacingMode === "environment") {
    switchCameraBtn.textContent = "Rear Camera";
  } else if (activeFacingMode === "user") {
    switchCameraBtn.textContent = "Front Camera";
  } else {
    switchCameraBtn.textContent = "Switch Camera";
  }

  switchCameraBtn.title = enabled
    ? `Currently using ${switchCameraBtn.textContent.toLowerCase()}`
    : "Only one camera detected";
}

function updateCameraAspectRatio() {
  if (!cameraCardEl) {
    return;
  }
  if (!videoEl.videoWidth || !videoEl.videoHeight) {
    return;
  }
  cameraCardEl.style.setProperty("--camera-aspect-ratio", `${videoEl.videoWidth} / ${videoEl.videoHeight}`);
}

async function refreshVideoDevices() {
  if (!navigator.mediaDevices?.enumerateDevices) {
    cameraDevices = [];
    updateSwitchCameraButton();
    return;
  }

  const devices = await navigator.mediaDevices.enumerateDevices();
  cameraDevices = devices.filter((d) => d.kind === "videoinput");
  updateSwitchCameraButton();
}

function stopCameraStream() {
  if (stream) {
    for (const track of stream.getTracks()) {
      track.stop();
    }
  }
  stream = null;
  videoEl.srcObject = null;
}

function stopFrameLoop() {
  if (animationFrameId !== null) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }
}

function startFrameLoop() {
  stopFrameLoop();
  const loop = async () => {
    animationFrameId = requestAnimationFrame(loop);

    if (!hands || !videoEl.srcObject || isSendingFrame || videoEl.readyState < 2) {
      return;
    }

    isSendingFrame = true;
    try {
      await hands.send({ image: videoEl });
    } finally {
      isSendingFrame = false;
    }
  };

  animationFrameId = requestAnimationFrame(loop);
}

async function startVideoStream() {
  stopCameraStream();
  resetFrameTracking();

  const videoConstraint = {
    width: { ideal: 1280 },
    height: { ideal: 720 },
  };

  if (activeDeviceId) {
    videoConstraint.deviceId = { exact: activeDeviceId };
  } else {
    videoConstraint.facingMode = { ideal: activeFacingMode };
  }

  stream = await navigator.mediaDevices.getUserMedia({
    video: videoConstraint,
    audio: false,
  });

  videoEl.srcObject = stream;
  await videoEl.play();
  updateCameraAspectRatio();

  const track = stream.getVideoTracks()[0];
  const settings = track ? track.getSettings() : null;
  if (settings) {
    activeDeviceId = settings.deviceId || activeDeviceId;
    if (settings.facingMode === "environment" || settings.facingMode === "user") {
      activeFacingMode = settings.facingMode;
    } else {
      const inferredFromTrack = inferFacingModeFromLabel(track?.label);
      if (inferredFromTrack) {
        activeFacingMode = inferredFromTrack;
      }
    }
  }

  await refreshVideoDevices();
  const activeDevice = cameraDevices.find((d) => d.deviceId === activeDeviceId);
  const inferredFromDevice = inferFacingModeFromLabel(activeDevice?.label);
  if (inferredFromDevice) {
    activeFacingMode = inferredFromDevice;
  }
  updateSwitchCameraButton();
}

function setStatus(text, kind = "") {
  if (!statusEl) {
    return;
  }
  statusEl.textContent = text;
  statusEl.classList.remove("ok", "error");
  if (kind) {
    statusEl.classList.add(kind);
  }
}

function setMode(mode) {
  if (!modeBadgeEl) {
    return;
  }
  modeBadgeEl.textContent = mode;
  modeBadgeEl.classList.toggle("gesture", mode === "GESTURE");
}

function setGestureHint(text) {
  if (!gestureHintEl) {
    return;
  }
  gestureHintEl.textContent = text;
}

function renderWord() {
  currentWordEl.textContent = currentWord || "(none)";
}

function renderTranscript() {
  transcriptEl.textContent = transcriptWords.length ? transcriptWords.join(" ") : "(empty)";
}

function renderLetter(letter, confidence, progress = 0) {
  liveLetterEl.textContent = letter;
  if (confidence > 0) {
    liveConfidenceEl.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
  } else {
    liveConfidenceEl.textContent = "Confidence: --";
  }
  holdProgressEl.style.width = `${Math.max(0, Math.min(progress, 1)) * 100}%`;
}

function renderSuggestions() {
  if (!suggestionsEl) {
    return;
  }
  suggestionsEl.replaceChildren();

  if (!suggestions.length) {
    const none = document.createElement("p");
    none.className = "suggestion-empty";
    none.textContent = "No suggestions";
    suggestionsEl.appendChild(none);
    return;
  }

  suggestions.forEach((word, idx) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `suggestion-chip${idx === 0 ? " top" : ""}`;
    btn.textContent = word;
    btn.addEventListener("click", () => {
      acceptSuggestionAt(idx);
    });
    suggestionsEl.appendChild(btn);
  });
}

function updateSuggestions() {
  suggestions = computeSmartPredictions(currentWord, transcriptWords, SUGGESTION_TOP_K);
  renderSuggestions();
}

function createEmptyVocabulary() {
  return {
    core_words: {},
    user_words: {},
    ngrams: {},
    stats: {
      created: Date.now() / 1000,
      source: "web",
    },
  };
}

function normalizeVocabulary(raw) {
  const base = createEmptyVocabulary();
  if (!raw || typeof raw !== "object") {
    return base;
  }

  if (raw.core_words && typeof raw.core_words === "object") {
    base.core_words = raw.core_words;
  }
  if (raw.user_words && typeof raw.user_words === "object") {
    base.user_words = raw.user_words;
  }
  if (raw.ngrams && typeof raw.ngrams === "object") {
    base.ngrams = raw.ngrams;
  }
  if (raw.stats && typeof raw.stats === "object") {
    base.stats = raw.stats;
  }

  return base;
}

function ensureCoreDefaults() {
  for (const word of CORE_DEFAULT_WORDS) {
    if (!vocabMemory.core_words[word]) {
      vocabMemory.core_words[word] = { source: "default" };
    }
  }
}

function saveVocabularyMemory() {
  try {
    localStorage.setItem(VOCAB_STORAGE_KEY, JSON.stringify(vocabMemory));
  } catch (err) {
    console.warn("Could not save vocabulary memory:", err);
  }
}

async function loadVocabularyMemory() {
  let loaded = null;

  try {
    const cached = localStorage.getItem(VOCAB_STORAGE_KEY);
    if (cached) {
      loaded = JSON.parse(cached);
    }
  } catch (err) {
    console.warn("Failed to parse local vocabulary cache:", err);
  }

  if (!loaded) {
    try {
      const response = await fetch(VOCAB_SEED_URL, { cache: "no-cache" });
      if (response.ok) {
        loaded = await response.json();
      }
    } catch (err) {
      console.warn("Failed to load seed vocabulary:", err);
    }
  }

  vocabMemory = normalizeVocabulary(loaded);
  ensureCoreDefaults();
  saveVocabularyMemory();
}

function registerWord(word) {
  const cleanWord = String(word || "").trim().toUpperCase();
  if (!cleanWord) {
    return;
  }

  const now = Date.now() / 1000;
  const existing = vocabMemory.user_words[cleanWord];
  if (!existing || typeof existing !== "object") {
    vocabMemory.user_words[cleanWord] = {
      frequency: 1,
      last_used: now,
    };
  } else {
    existing.frequency = Number(existing.frequency || 0) + 1;
    existing.last_used = now;
  }
}

function registerSequence(historyList) {
  if (!Array.isArray(historyList) || historyList.length < 2) {
    return;
  }

  const targetWord = historyList[historyList.length - 1];
  for (let i = 1; i < VOCAB_N_ORDER; i += 1) {
    if (historyList.length < i + 1) {
      break;
    }

    const context = historyList.slice(-(i + 1), -1).join(" ");
    if (!vocabMemory.ngrams[context]) {
      vocabMemory.ngrams[context] = {};
    }

    const prev = Number(vocabMemory.ngrams[context][targetWord] || 0);
    vocabMemory.ngrams[context][targetWord] = prev + 1;
  }
}

function computeSmartPredictions(prefixRaw, historyRaw, topK = 3) {
  const prefix = String(prefixRaw || "").toUpperCase();
  const history = Array.isArray(historyRaw)
    ? historyRaw.map((w) => String(w || "").trim().toUpperCase()).filter(Boolean)
    : [];

  const candidates = new Map();
  const now = Date.now() / 1000;

  const getHist = (idx) => {
    if (history.length >= Math.abs(idx)) {
      return history[history.length + idx];
    }
    return null;
  };

  const addScore = (wordRaw, basePoints) => {
    const word = String(wordRaw || "").toUpperCase();
    if (!word) {
      return;
    }
    if (prefix && !word.startsWith(prefix)) {
      return;
    }

    let total = Number(basePoints || 0);
    const user = vocabMemory.user_words[word];
    if (user && (now - Number(user.last_used || 0)) < 300) {
      total += 30;
    }

    if (getHist(-1) === word) {
      total -= 500;
    }
    if (getHist(-2) === word) {
      total -= 50;
    }

    candidates.set(word, (candidates.get(word) || 0) + total);
  };

  let contextFound = false;
  for (let i = VOCAB_N_ORDER - 1; i > 0; i -= 1) {
    if (history.length < i) {
      continue;
    }

    const context = history.slice(-i).join(" ");
    const nextMap = vocabMemory.ngrams[context];
    if (!nextMap || typeof nextMap !== "object") {
      continue;
    }

    const tierScore = 100 * i;
    for (const [nextWord, countRaw] of Object.entries(nextMap)) {
      const count = Number(countRaw || 0);
      const freqBoost = Math.log(count + 1) * 10;
      addScore(nextWord, tierScore + freqBoost);
      contextFound = true;
    }
  }

  const shouldShowGenerics = prefix !== "" || !contextFound;
  if (shouldShowGenerics) {
    const allWords = new Set([
      ...Object.keys(vocabMemory.user_words || {}),
      ...Object.keys(vocabMemory.core_words || {}),
    ]);

    for (const word of allWords) {
      if (candidates.has(word) && candidates.get(word) > 50) {
        continue;
      }
      if (prefix && !word.startsWith(prefix)) {
        continue;
      }

      const user = vocabMemory.user_words[word];
      const freq = user ? Number(user.frequency || 0) : 0;
      addScore(word, Math.log(freq + 1) * 2);
    }
  }

  if (prefix && candidates.has(prefix)) {
    candidates.set(prefix, candidates.get(prefix) + 20);
  }

  return [...candidates.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, topK)
    .map(([word]) => word);
}

function speakText(text) {
  if (!text || !window.speechSynthesis) {
    return;
  }

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1;
  utterance.pitch = 1;
  window.speechSynthesis.speak(utterance);
}

function resetCurrentWordState() {
  letterBuffer = [];
  currentWord = "";
  renderWord();
}

function finalizeWord(rawWord, sourceTag = "WORD", shouldSpeak = true) {
  const word = String(rawWord || "").trim().toUpperCase();
  if (!word) {
    return;
  }

  transcriptWords.push(word);
  registerWord(word);
  registerSequence(transcriptWords);
  saveVocabularyMemory();

  resetCurrentWordState();
  renderTranscript();
  updateSuggestions();
  setGestureHint(`${sourceTag}: ${word}`);

  if (shouldSpeak) {
    speakText(word);
  }
}

function confirmWordAction() {
  if (currentWord) {
    finalizeWord(currentWord, "RAW INPUT");
    return;
  }

  if (suggestions.length) {
    finalizeWord(suggestions[0], "CONFIRMED");
  }
}

function acceptTopSuggestionAction() {
  if (suggestions.length) {
    finalizeWord(suggestions[0], "AUTO-COMPLETE");
    return;
  }

  if (currentWord) {
    finalizeWord(currentWord, "RAW INPUT");
  }
}

function acceptSuggestionAt(index) {
  const word = suggestions[index];
  if (!word) {
    return;
  }
  finalizeWord(word, "SELECTED");
}

function backspaceLetter() {
  if (!letterBuffer.length) {
    return;
  }

  letterBuffer.pop();
  currentWord = letterBuffer.join("");
  renderWord();
  updateSuggestions();
  setGestureHint("Edited current word");
}

function undoLastWord() {
  if (!transcriptWords.length) {
    return;
  }

  const removed = transcriptWords.pop();
  renderTranscript();
  updateSuggestions();
  setGestureHint(`UNDO: ${removed}`);
}

function clearWord() {
  if (!currentWord) {
    return;
  }

  resetCurrentWordState();
  updateSuggestions();
  setGestureHint("Cleared current word");
}

function speakCurrentWord() {
  const text = currentWord || transcriptWords.join(" ");
  if (!text) {
    return;
  }
  speakText(text);
}

function handleGestureAction(gesture) {
  setMode("GESTURE");

  if (gesture === "THUMB_UP") {
    confirmWordAction();
    return;
  }

  if (gesture === "SMART_SELECT") {
    acceptTopSuggestionAction();
    return;
  }

  if (gesture === "THUMB_DOWN") {
    if (currentWord) {
      clearWord();
      setGestureHint("Gesture THUMB_DOWN: clear word");
    } else {
      undoLastWord();
      setGestureHint("Gesture THUMB_DOWN: undo word");
    }
    return;
  }

  if (gesture === "PINCH") {
    backspaceLetter();
    setGestureHint("Gesture PINCH: backspace");
    return;
  }

  if (gesture === "OPEN_PALM") {
    if (transcriptWords.length) {
      speakText(transcriptWords.join(" "));
    }
    setGestureHint("Gesture OPEN_PALM: sentence completed");
  }
}

class GestureController {
  constructor(bufferSize = 10, cooldownFrames = 20) {
    this.bufferSize = bufferSize;
    this.cooldownFrames = cooldownFrames;
    this.gestureBuffer = [];
    this.cooldownCounter = 0;
    this.validGestures = new Set([
      "THUMB_UP",
      "THUMB_DOWN",
      "PINCH",
      "OPEN_PALM",
      "SMART_SELECT",
    ]);
  }

  updateAndCheck(landmarks) {
    const gesture = this.detectFrameGesture(landmarks);

    this.gestureBuffer.push(gesture);
    if (this.gestureBuffer.length > this.bufferSize) {
      this.gestureBuffer.shift();
    }

    if (this.cooldownCounter > 0) {
      this.cooldownCounter -= 1;
      return null;
    }

    const required = Math.floor(this.bufferSize * 0.7);
    for (const g of this.validGestures) {
      const count = this.gestureBuffer.filter((value) => value === g).length;
      if (count >= required) {
        this.cooldownCounter = this.cooldownFrames;
        this.gestureBuffer = [];
        return g;
      }
    }

    return null;
  }

  isPotentialGesture() {
    const count = this.gestureBuffer.filter((g) => this.validGestures.has(g)).length;
    return count >= Math.floor(this.bufferSize * 0.4);
  }

  detectFrameGesture(lm) {
    if (!Array.isArray(lm) || lm.length < 21) {
      return null;
    }

    const WRIST = 0;
    const THUMB_TIP = 4;
    const INDEX_MCP = 5;
    const INDEX_TIP = 8;
    const MID_MCP = 9;
    const MID_TIP = 12;
    const RING_MCP = 13;
    const RING_TIP = 16;
    const PINKY_MCP = 17;
    const PINKY_TIP = 20;

    const dist = (i, j) => distance(lm[i], lm[j]);
    const palm = dist(WRIST, MID_MCP) + 1e-6;

    if (
      dist(THUMB_TIP, INDEX_MCP) > palm * 0.4
      && dist(INDEX_TIP, WRIST) > dist(INDEX_MCP, WRIST) * 1.5
      && dist(PINKY_TIP, WRIST) > dist(PINKY_MCP, WRIST) * 1.5
      && dist(MID_TIP, WRIST) < dist(MID_MCP, WRIST) * 1.2
      && dist(RING_TIP, WRIST) < dist(RING_MCP, WRIST) * 1.2
    ) {
      return "SMART_SELECT";
    }

    let fingersFolded = true;
    for (const [tip, mcp] of [[INDEX_TIP, INDEX_MCP], [MID_TIP, MID_MCP], [RING_TIP, RING_MCP], [PINKY_TIP, PINKY_MCP]]) {
      if (dist(tip, WRIST) > dist(mcp, WRIST)) {
        fingersFolded = false;
        break;
      }
    }

    if (fingersFolded) {
      if (dist(THUMB_TIP, INDEX_MCP) > palm * 0.6) {
        if (lm[THUMB_TIP][1] < lm[WRIST][1] - palm * 0.3) {
          return "THUMB_UP";
        }
        if (lm[THUMB_TIP][1] > lm[WRIST][1] + palm * 0.3) {
          return "THUMB_DOWN";
        }
      }
    }

    if (!fingersFolded) {
      if (dist(THUMB_TIP, INDEX_TIP) < palm * 0.15) {
        const foldedRest = [MID_TIP, RING_TIP, PINKY_TIP].every(
          (tip) => dist(tip, WRIST) < dist(MID_MCP, WRIST) * 1.3,
        );

        if (foldedRest) {
          return "PINCH";
        }
      }

      const openPalm = [THUMB_TIP, INDEX_TIP, MID_TIP, RING_TIP, PINKY_TIP].every(
        (tip) => dist(tip, WRIST) > palm * 0.85,
      );

      if (openPalm && dist(INDEX_TIP, PINKY_TIP) > palm * 0.9) {
        return "OPEN_PALM";
      }
    }

    return null;
  }
}

function vecSub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function vecScale(v, s) {
  return [v[0] * s, v[1] * s, v[2] * s];
}

function vecDot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function vecNorm(v) {
  return Math.hypot(v[0], v[1], v[2]);
}

function vecCross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function distance(a, b) {
  return vecNorm(vecSub(a, b));
}

function angle(a, b, c, eps = 1e-8) {
  const ba = vecSub(a, b);
  const bc = vecSub(c, b);
  const baNorm = vecNorm(ba) + eps;
  const bcNorm = vecNorm(bc) + eps;
  const baUnit = vecScale(ba, 1 / baNorm);
  const bcUnit = vecScale(bc, 1 / bcNorm);
  const cosVal = Math.max(-1, Math.min(1, vecDot(baUnit, bcUnit)));
  return Math.acos(cosVal);
}

function featurizePose(flat63) {
  const pts = [];
  for (let i = 0; i < 21; i += 1) {
    pts.push([
      flat63[i * 3],
      flat63[i * 3 + 1],
      flat63[i * 3 + 2],
    ]);
  }

  const wrist = pts[0];
  for (let i = 0; i < pts.length; i += 1) {
    pts[i] = vecSub(pts[i], wrist);
  }

  const scale = vecNorm(pts[9]) + 1e-6;
  for (let i = 0; i < pts.length; i += 1) {
    pts[i] = vecScale(pts[i], 1 / scale);
  }

  let normal = vecCross(pts[5], pts[17]);
  normal = vecScale(normal, 1 / (vecNorm(normal) + 1e-6));

  const tipIdx = [4, 8, 12, 16, 20];
  const tipD = tipIdx.map((idx) => vecNorm(pts[idx]));

  const tips = [pts[8], pts[12], pts[16], pts[20]];
  const inter = [];
  for (let i = 0; i < tips.length; i += 1) {
    for (let j = i + 1; j < tips.length; j += 1) {
      inter.push(distance(tips[i], tips[j]));
    }
  }

  const angTriples = [
    [5, 6, 7],
    [9, 10, 11],
    [13, 14, 15],
    [17, 18, 19],
  ];
  const angs = angTriples.map(([a, b, c]) => angle(pts[a], pts[b], pts[c]));

  const thumb = pts[4];
  const index = pts[5];
  const middle = pts[9];

  const thumbToIndex = distance(thumb, index);
  const thumbToMiddle = distance(thumb, middle);
  const thumbFeatures = [
    thumbToIndex,
    thumbToMiddle,
    thumbToMiddle - thumbToIndex,
  ];

  const features = [];
  for (let i = 0; i < pts.length; i += 1) {
    features.push(pts[i][0], pts[i][1], pts[i][2]);
  }
  features.push(normal[0], normal[1], normal[2]);
  features.push(...tipD, ...inter, ...angs, ...thumbFeatures);

  return Float32Array.from(features);
}

function averageFeatures(buffer) {
  const out = new Float32Array(FEATURE_DIM);
  for (const f of buffer) {
    for (let i = 0; i < FEATURE_DIM; i += 1) {
      out[i] += f[i];
    }
  }
  for (let i = 0; i < FEATURE_DIM; i += 1) {
    out[i] /= buffer.length;
  }
  return out;
}

function argmax(values) {
  let bestIdx = 0;
  let bestVal = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < values.length; i += 1) {
    if (values[i] > bestVal) {
      bestVal = values[i];
      bestIdx = i;
    }
  }
  return bestIdx;
}

function mode(values) {
  const counts = new Map();
  let bestValue = values[0];
  let bestCount = 0;

  for (const value of values) {
    const next = (counts.get(value) || 0) + 1;
    counts.set(value, next);
    if (next > bestCount) {
      bestCount = next;
      bestValue = value;
    }
  }

  return bestValue;
}

async function loadLabels() {
  try {
    const res = await fetch("./labels.json", { cache: "no-cache" });
    if (!res.ok) {
      return;
    }
    const json = await res.json();
    if (Array.isArray(json) && json.length) {
      labels = json.map((x) => String(x));
    }
  } catch {
    // Keep defaults if labels are missing.
  }
}

async function loadModel() {
  await loadLabels();
  model = await tf.loadLayersModel("./model/model.json", {
    strict: false,
  });

  const inShape = model.inputs?.[0]?.shape || [];
  const outShape = model.outputs?.[0]?.shape || [];
  const inputDim = inShape[inShape.length - 1];
  const classCount = outShape[outShape.length - 1];

  if (inputDim !== FEATURE_DIM) {
    throw new Error(`Expected model input ${FEATURE_DIM}, got ${inputDim}`);
  }

  if (labels.length !== classCount) {
    throw new Error(`Label count ${labels.length} does not match model classes ${classCount}`);
  }
}

function drawScene(results) {
  const image = results.image;
  if (!image) {
    return;
  }

  if (canvasEl.width !== image.width || canvasEl.height !== image.height) {
    canvasEl.width = image.width;
    canvasEl.height = image.height;
  }

  ctx.save();
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

  if (UNMIRROR_PREVIEW) {
    // Some front cameras deliver a mirrored stream. Flip once to show true orientation.
    ctx.translate(canvasEl.width, 0);
    ctx.scale(-1, 1);
  }

  ctx.drawImage(image, 0, 0, canvasEl.width, canvasEl.height);

  if (results.multiHandLandmarks && results.multiHandLandmarks.length) {
    const landmarks = results.multiHandLandmarks[0];
    drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
      color: "#41e2bf",
      lineWidth: 4,
    });
    drawLandmarks(ctx, landmarks, {
      color: "#f5f5f5",
      lineWidth: 1,
      radius: 2,
    });
  }

  ctx.restore();
}

function resetFrameTracking() {
  rawPredBuffer = [];
  featureBuffer = [];
  lastDetectedLetter = null;
  letterHoldFrames = 0;
}

function handlePrediction(landmarks) {
  const flat = [];
  for (const lm of landmarks) {
    flat.push(lm.x, lm.y, lm.z);
  }

  const features = featurizePose(flat);
  featureBuffer.push(features);
  if (featureBuffer.length > FEATURE_BUFFER_SIZE) {
    featureBuffer.shift();
  }

  if (featureBuffer.length === 0) {
    return;
  }

  const avg = averageFeatures(featureBuffer);
  const prediction = tf.tidy(() => {
    const input = tf.tensor2d(avg, [1, FEATURE_DIM]);
    const out = model.predict(input);
    return out.dataSync();
  });

  const idx = argmax(prediction);
  const conf = prediction[idx];
  const candidate = labels[idx] || "?";
  const requiredConf = CONFIDENCE_THRESHOLDS[candidate] || CONFIDENCE_THRESHOLDS.default;

  if (conf < requiredConf) {
    renderLetter("_", 0, 0);
    return;
  }

  rawPredBuffer.push(idx);
  if (rawPredBuffer.length > RAW_PRED_BUFFER_SIZE) {
    rawPredBuffer.shift();
  }

  if (rawPredBuffer.length < 3) {
    renderLetter(candidate, conf, 0);
    return;
  }

  const stableIdx = mode(rawPredBuffer);
  const stableLetter = labels[stableIdx] || "?";

  if (stableLetter === lastDetectedLetter) {
    letterHoldFrames += 1;
  } else {
    lastDetectedLetter = stableLetter;
    letterHoldFrames = 1;
  }

  let threshold = conf > 0.85 ? FAST_MODE_THRESHOLD : LETTER_HOLD_THRESHOLD;
  if (letterBuffer.length && stableLetter === letterBuffer[letterBuffer.length - 1]) {
    threshold = REPEAT_DELAY_FRAMES;
  }

  if (letterHoldFrames >= threshold) {
    letterBuffer.push(stableLetter);
    currentWord = letterBuffer.join("");
    renderWord();
    updateSuggestions();
    letterHoldFrames = 0;
  }

  renderLetter(stableLetter, conf, letterHoldFrames / threshold);
}

function wireEvents() {
  if (confirmWordBtn) {
    confirmWordBtn.addEventListener("click", confirmWordAction);
  }
  if (acceptSuggestionBtn) {
    acceptSuggestionBtn.addEventListener("click", acceptTopSuggestionAction);
  }
  if (backspaceBtn) {
    backspaceBtn.addEventListener("click", backspaceLetter);
  }
  if (undoWordBtn) {
    undoWordBtn.addEventListener("click", undoLastWord);
  }
  if (clearWordBtn) {
    clearWordBtn.addEventListener("click", clearWord);
  }
  if (speakWordBtn) {
    speakWordBtn.addEventListener("click", speakCurrentWord);
  }
  if (switchCameraBtn) {
    switchCameraBtn.addEventListener("click", async () => {
      if (isSwitchingCamera) {
        return;
      }

      isSwitchingCamera = true;
      switchCameraBtn.disabled = true;

      try {
        setStatus("Switching camera...", "ok");
        await refreshVideoDevices();

        if (cameraDevices.length >= 2) {
          const idx = cameraDevices.findIndex((d) => d.deviceId === activeDeviceId);
          const nextIdx = idx >= 0 ? (idx + 1) % cameraDevices.length : 0;
          activeDeviceId = cameraDevices[nextIdx].deviceId;
        } else {
          activeDeviceId = null;
          activeFacingMode = activeFacingMode === "environment" ? "user" : "environment";
        }

        await startVideoStream();
        setStatus("Live recognition + predictor running", "ok");
      } catch (err) {
        console.error(err);
        setStatus(`Camera switch failed: ${err.message}`, "error");
      } finally {
        isSwitchingCamera = false;
        updateSwitchCameraButton();
      }
    });
  }

  document.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      confirmWordAction();
      event.preventDefault();
      return;
    }

    if (event.key === "Tab") {
      acceptTopSuggestionAction();
      event.preventDefault();
      return;
    }

    if (event.key === "Backspace") {
      if (event.ctrlKey || event.metaKey) {
        undoLastWord();
      } else {
        backspaceLetter();
      }
      event.preventDefault();
      return;
    }

    if (event.key === "Escape") {
      clearWord();
    }
  });
}

async function startCameraLoop() {
  hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
  });

  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  hands.onResults((results) => {
    drawScene(results);

    if (spellingCooldownFrames > 0) {
      spellingCooldownFrames -= 1;
    }

    const hasHand = results.multiHandLandmarks && results.multiHandLandmarks.length > 0;
    if (!hasHand) {
      resetFrameTracking();
      renderLetter("_", 0, 0);
      setMode("SPELLING");
      setGestureHint("Gesture: none");
      return;
    }

    const landmarks = results.multiHandLandmarks[0];
    const landmarkList = landmarks.map((lm) => [lm.x, lm.y, lm.z]);

    const gesture = gestureCtrl.updateAndCheck(landmarkList);
    if (gesture) {
      spellingCooldownFrames = SPELLING_COOLDOWN_DURATION;
      resetFrameTracking();
      handleGestureAction(gesture);
      return;
    }

    const cmdLock = gestureCtrl.isPotentialGesture() && currentWord === "";
    if (cmdLock || spellingCooldownFrames > 0) {
      setMode("GESTURE");
      if (cmdLock) {
        setGestureHint("Gesture ready (CMD LOCK)");
      }
      return;
    }

    setMode("SPELLING");
    handlePrediction(landmarks);
  });

  videoEl.addEventListener("loadedmetadata", updateCameraAspectRatio);
  await startVideoStream();
  startFrameLoop();
}

async function bootstrap() {
  wireEvents();
  renderWord();
  renderTranscript();
  renderLetter("_", 0, 0);
  setMode("SPELLING");
  setGestureHint("Gesture: none");
  gestureCtrl = new GestureController();

  try {
    setStatus("Loading vocabulary memory...");
    await loadVocabularyMemory();
    updateSuggestions();

    setStatus("Loading TensorFlow.js model...");
    await loadModel();

    setStatus("Starting webcam...", "ok");
    await refreshVideoDevices();
    await startCameraLoop();
    setStatus("Live recognition + predictor running", "ok");
  } catch (err) {
    console.error(err);
    setStatus(`Startup failed: ${err.message}`, "error");
  }
}

window.addEventListener("beforeunload", () => {
  stopFrameLoop();
  stopCameraStream();
});

bootstrap();
