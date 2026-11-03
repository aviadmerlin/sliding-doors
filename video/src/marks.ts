// Timestamps (seconds, relative to the start of each clip) captured by
// scripts/record-footage.mjs while recording the live app. Copied from
// public/recordings/marks.json -- see that script for how each beat was
// produced and public/recordings/timestamps.json for the raw capture.
export const desktopMarks = {
  pageLoaded: 5.261,
  scoreboardVisible: 6.586,
  winProbAndTurningPointsVisible: 12.585,
  whatifSectionVisible: 29.317,
  eventDropdownOpen: 30.918,
  event84Selected: 31.845,
  runCounterfactualClicked: 31.873,
  counterfactualResultVisible: 38.876,
  agentSectionVisible: 45.935,
  enQuestionTypingStart: 47.407,
  enQuestionTypingEnd: 50.479,
  enAskClicked: 51.019,
  enAnswerVisible: 67.022,
  heQuestionTypingStart: 72.405,
  heQuestionTypingEnd: 74.894,
  heAskClicked: 75.435,
  heAnswerVisible: 93.436,
  refusalQuestionTypingStart: 98.801,
  refusalQuestionTypingEnd: 101.119,
  refusalAskClicked: 101.65,
  refusalAnswerVisible: 117.651,
  clubProfileVisible: 124.498,
  recordingEnd: 131.5,
} as const;

export const mobileMarks = {
  heroVisible: 4.5,
  scrollDone: 8.558,
  recordingEnd: 10.06,
} as const;
