const translations = {
  zh: {
    navOverview: "概览",
    navMethod: "方法",
    navResults: "结果",
    navDynamics: "训练动态",
    navPaper: "论文",
    languageButton: "EN",
    heroEyebrow: "弱到强泛化",
    heroTitle: "通过 Direct On-Policy Distillation 实现弱到强泛化",
    heroSubtitle:
      "Direct-OPD 迁移小教师模型在 RL 中学到的策略变化，而不是模仿它的最终分布。这个变化提供稠密的 token 级监督，并在强学生模型自己的 on-policy 状态上发挥作用。",
    readPdf: "阅读论文",
    code: "代码",
    bibtex: "引用",
    computeLabel: "迁移成本",
    compositionLabel: "顺序策略迁移",
    heroCaption:
      "Direct-OPD 迁移 RL 学到的改进方向；普通 endpoint OPD 则会把强学生拉向较弱教师的能力上限。",
    citationEyebrow: "引用",
    methodEyebrow: "核心机制",
    methodTitle: "迁移策略变化，而不是教师模型的能力上限。",
    methodIntro:
      "小教师的参考模型和 RL 后模型共同编码了 RL 发现的奖励。Direct-OPD 将这个奖励读成 log-ratio，并应用在强学生模型采样出的前缀上。",
    stepOneTitle: "RL 前参考模型",
    stepOneText: "从小教师 RL 之前的模型开始：pi_Tref。",
    stepTwoTitle: "RL 后教师模型",
    stepTwoText: "RL 将小模型推向更好的推理行为：pi_T。",
    stepThreeTitle: "稠密隐式奖励",
    stepThreeText: "用 log pi_T - log pi_Tref 作为强学生的 token 级改进方向。",
    endpointLabel: "Endpoint OPD",
    endpointText: "复制弱教师的最终分布。",
    directText: "把教师的 RL 改进方向应用到强学生模型。",
    problemEyebrow: "为什么普通 OPD 会失败",
    problemTitle: "Endpoint imitation 混合了两种信号。",
    problemText:
      "普通 OPD 要求学生匹配 RL 后教师模型的分布。在弱到强迁移里，这个分布同时包含教师有用的 RL 改进，以及小模型自身的能力上限。因此，一个更强的学生可能被拉低到弱教师的能力天花板。",
    signalEyebrow: "Direct-OPD 迁移什么",
    signalTitle: "教师模型由 RL 诱导出的策略变化",
    signalText:
      "Direct-OPD 用教师的 RL 后 checkpoint 减去它自己的 RL 前参考模型。这一步隔离出 RL 改变了什么，同时丢弃教师的绝对 endpoint policy。",
    objectiveEyebrow: "训练目标",
    objectiveTitle: "在学生访问的状态上使用稠密奖励",
    objectiveText:
      "学生从当前策略采样自己的 rollout。对每个访问到的前缀，Direct-OPD 在学生 top-k 候选 token 上计算教师/参考模型的 log-ratio，提高 RL 后教师变得更偏好的 token 概率，降低被 RL 抑制的 token 概率。",
    objStepOneTitle: "从学生模型采样",
    objStepOneText: "使用当前学生的 on-policy 前缀，而不是教师生成的轨迹。",
    objStepTwoTitle: "给 top-k 动作打分",
    objStepTwoText: "只在学生实际考虑的候选 token 上读取 log pi_T - log pi_Tref。",
    objStepThreeTitle: "用 KL anchor 更新",
    objStepThreeText: "让学生保持接近初始化模型，使稠密奖励保持可靠。",
    compareAxis: "维度",
    compareTarget: "目标",
    compareSignal: "监督信号",
    compareRisk: "弱到强风险",
    compareOpdTarget: "教师 endpoint 分布",
    compareDirectTarget: "教师/参考模型策略变化",
    compareOpdSignal: "log pi_T - log pi_student",
    compareDirectSignal: "log pi_T - log pi_Tref",
    compareOpdRisk: "可能引入弱教师的能力上限",
    compareDirectRisk: "保留学生更强的基座能力，同时迁移改进方向",
    resultsEyebrow: "实验结果",
    resultsTitle: "更弱的 RL 教师可以提升更强的学生。",
    resultsIntro:
      "在 JustRL 和 QuestA 两组教师信号上，Direct-OPD 都能提升不同模型家族的学生，包括已经强于 RL 后教师的学生。",
    transferCaption:
      "JustRL 策略变化迁移在 AIME 2024 和 AIME 2025 上提升 Qwen3-1.7B、Qwen3-4B 和 R1-Distill-7B。",
    galleryEyebrow: "可视化展示",
    galleryTitle: "结果、计算成本、组合能力和训练动态。",
    tabCompute: "计算成本",
    tabComposition: "组合",
    tabLength: "长度",
    tabKl: "KL",
    computeCaption:
      "先在小模型上运行 RL，再迁移策略变化；在相同计算预算下，这条 weak-to-strong 路线优于直接在大模型上跑 RL。",
    compositionCaption:
      "JustRL 和 QuestA 两个独立策略变化可以顺序应用到同一个学生上，在 AIME 2024 达到 63.8。",
    lengthCaption:
      "短 horizon 的 Direct-OPD 训练可以改变更长 rollout 的行为，同时避开不可靠的后段前缀信号。",
    klCaption:
      "Adaptive KL 让教师-参考模型的稠密奖励保持在信息有效区间，而不是盲目最大化该奖励。",
    paperEyebrow: "论文",
    paperTitle: "摘要和贡献",
    abstractTitle: "摘要",
    abstractText:
      "基于可验证奖励的强化学习能提升推理能力，但依赖稀疏的终局奖励。Direct-OPD 使用一个小模型的 RL 后教师和 RL 前参考模型来恢复稠密 token 级隐式奖励，让更强的学生模型无需大规模目标模型 RL，也能在自己的 on-policy 状态上继续提升。",
    contribTitle: "贡献",
    contribOne: "从 credit assignment 角度解释弱到强迁移。",
    contribTwo: "在不同教师对、学生模型和模型家族上获得稳定提升。",
    contribThree: "相比直接大模型 RL，具备更好的计算成本收益。",
    contribFour: "分析 overlap、response length 和 KL 对信号可靠性的影响。",
    footerPdf: "论文 PDF",
    backTop: "回到顶部"
  }
};

const english = {};
document.querySelectorAll("[data-i18n]").forEach((node) => {
  english[node.dataset.i18n] = node.textContent;
});

let currentLang = "en";

function setLanguage(lang) {
  currentLang = lang;
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    const key = node.dataset.i18n;
    node.textContent = lang === "zh" ? translations.zh[key] || english[key] : english[key];
  });
}

document.querySelector("[data-lang-toggle]").addEventListener("click", () => {
  setLanguage(currentLang === "en" ? "zh" : "en");
});

document.querySelector("[data-bib-toggle]").addEventListener("click", () => {
  const panel = document.querySelector("[data-bib-panel]");
  panel.hidden = !panel.hidden;
  if (!panel.hidden) {
    panel.scrollIntoView({ behavior: "smooth", block: "center" });
  }
});

document.querySelectorAll("[data-tab]").forEach((tab) => {
  tab.addEventListener("click", () => {
    const name = tab.dataset.tab;
    document.querySelectorAll("[data-tab]").forEach((item) => {
      item.classList.toggle("active", item === tab);
      item.setAttribute("aria-selected", item === tab ? "true" : "false");
    });
    document.querySelectorAll("[data-panel]").forEach((panel) => {
      panel.classList.toggle("active", panel.dataset.panel === name);
    });
  });
});
