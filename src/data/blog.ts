import worriesOnAi from "../content/notes/worries-on-ai.md?raw";
import contextualMultinomialBandits from "../content/notes/contextual-multinomial-bandits.md?raw";
import roleOfRlInLlmReasoning from "../content/notes/role-of-rl-in-llm-reasoning.md?raw";

export type BlogPost = {
  title: string;
  date: string;
  summary: string;
  slug: string;
  type: "markdown" | "pdf";
  href?: string;
  content?: string;
  tags?: string[];
};

export const blogPosts: BlogPost[] = [
  {
    title: "Understanding LLM Pre-Training and RL Fine-Tuning via Meta-Learning",
    date: "2026-06",
    summary:
      "My undergraduate thesis, written in Chinese, studying LLM pre-training and RL fine-tuning through a meta-learning perspective.",
    slug: "undergraduate-thesis-meta-learning",
    type: "pdf",
    href: "/notes/undergraduate-thesis/llm-pretraining-rl-finetuning-meta-learning.pdf",
    tags: ["thesis", "Chinese", "LLM"],
  },
  {
    title: "Worries on AI",
    date: "2026-05",
    summary:
      "A personal reflection on language models, scaling, reasoning, self-evolution, and the gap between carbon-based and silicon-based intelligence.",
    slug: "worries-on-ai",
    type: "markdown",
    content: worriesOnAi,
    tags: ["AI", "LLM", "future"],
  },
  {
    title: "The Role of RL in LLM Reasoning",
    date: "2026-01",
    summary:
      "A long-form discussion of whether RLVR sharpens existing reasoning behavior or discovers new reasoning capabilities in language models.",
    slug: "role-of-rl-in-llm-reasoning",
    type: "markdown",
    content: roleOfRlInLlmReasoning,
    tags: ["RL", "LLM", "reasoning"],
  },
  {
    title: "Contextual Multinomial Bandits",
    date: "2025-08",
    summary:
      "A technical note on contextual multinomial logit bandits, optimistic algorithms, warm-up exploration, and regret analysis.",
    slug: "contextual-multinomial-bandits",
    type: "markdown",
    content: contextualMultinomialBandits,
    tags: ["bandits", "MNL", "theory"],
  },
];
