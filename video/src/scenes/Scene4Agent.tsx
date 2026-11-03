import { AbsoluteFill, Sequence } from "remotion";
import { Clip } from "../Clip";
import { VoiceOver } from "../VoiceOver";
import { desktopMarks } from "../marks";

const SPINNER_DURATION = 2.5;
const ANSWER_HOLD = 4.0;

const EN_LEAD = desktopMarks.enAskClicked - desktopMarks.agentSectionVisible;
const HE_LEAD = desktopMarks.heAskClicked - desktopMarks.heQuestionTypingStart;
const REFUSAL_LEAD =
  desktopMarks.refusalAskClicked - desktopMarks.refusalQuestionTypingStart;

const EN_TOTAL = EN_LEAD + SPINNER_DURATION + ANSWER_HOLD;
const HE_TOTAL = HE_LEAD + SPINNER_DURATION + ANSWER_HOLD;
const REFUSAL_TOTAL = REFUSAL_LEAD + SPINNER_DURATION + ANSWER_HOLD;

export const SCENE_4_DURATION_SEC = EN_TOTAL + HE_TOTAL + REFUSAL_TOTAL;

const Badge: React.FC<{ children: string }> = ({ children }) => (
  <div
    style={{
      position: "absolute",
      top: 120,
      right: 140,
      background: "rgba(8, 12, 20, 0.84)",
      border: "1px solid rgba(255,255,255,0.1)",
      borderRadius: 999,
      padding: "14px 30px",
      color: "#f2f4f8",
      fontFamily:
        "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
      fontSize: 32,
      fontWeight: 700,
      letterSpacing: 0.5,
    }}
  >
    {children}
  </div>
);

const QuestionBlock: React.FC<{
  fromSec: number;
  leadTrimBeforeSec: number;
  leadDuration: number;
  askClickedSec: number;
  answerVisibleSec: number;
  badge: string;
}> = ({ fromSec, leadTrimBeforeSec, leadDuration, askClickedSec, answerVisibleSec, badge }) => {
  const total = leadDuration + SPINNER_DURATION + ANSWER_HOLD;
  return (
    <Sequence from={Math.round(fromSec * 30)} durationInFrames={Math.round(total * 30)}>
      <Clip
        src="recordings/desktop.webm"
        fromSec={0}
        trimBeforeSec={leadTrimBeforeSec}
        sourceDurationSec={leadDuration}
      />
      <Clip
        src="recordings/desktop.webm"
        fromSec={leadDuration}
        trimBeforeSec={askClickedSec}
        sourceDurationSec={SPINNER_DURATION}
      />
      <Clip
        src="recordings/desktop.webm"
        fromSec={leadDuration + SPINNER_DURATION}
        trimBeforeSec={answerVisibleSec}
        sourceDurationSec={ANSWER_HOLD}
      />
      <Badge>{badge}</Badge>
    </Sequence>
  );
};

export const Scene4Agent: React.FC = () => {
  return (
    <AbsoluteFill style={{ background: "#05070c" }}>
      <QuestionBlock
        fromSec={0}
        leadTrimBeforeSec={desktopMarks.agentSectionVisible}
        leadDuration={EN_LEAD}
        askClickedSec={desktopMarks.enAskClicked}
        answerVisibleSec={desktopMarks.enAnswerVisible}
        badge="EN · grounding check"
      />
      <QuestionBlock
        fromSec={EN_TOTAL}
        leadTrimBeforeSec={desktopMarks.heQuestionTypingStart}
        leadDuration={HE_LEAD}
        askClickedSec={desktopMarks.heAskClicked}
        answerVisibleSec={desktopMarks.heAnswerVisible}
        badge="עברית · Hebrew"
      />
      <QuestionBlock
        fromSec={EN_TOTAL + HE_TOTAL}
        leadTrimBeforeSec={desktopMarks.refusalQuestionTypingStart}
        leadDuration={REFUSAL_LEAD}
        askClickedSec={desktopMarks.refusalAskClicked}
        answerVisibleSec={desktopMarks.refusalAnswerVisible}
        badge="Honest refusal"
      />
      <VoiceOver file="04-agent.mp3" />
    </AbsoluteFill>
  );
};
