export type Publication = {
  title: string;
  authors: string;
  venue: string;
  year: string;
  highlight?: string;
  note?: string;
  links?: {
    label: string;
    href: string;
  }[];
};

export const publications: Publication[] = [
  {
    title:
      "Q-MMR: Off-Policy Evaluation via Recursive Reweighting and Moment Matching",
    authors: "Xiang Li, Nan Jiang",
    venue: "arXiv preprint arXiv:2605.06474",
    year: "2026",
    highlight:
      "A new off-policy evaluation algorithm based on weight learning, with guarantees under general function approximation that require only function realizability.",
    links: [{ label: "paper", href: "https://arxiv.org/abs/2605.06474" }],
  },
  {
    title:
      "Beyond State-Wise Mirror Descent: Offline Policy Optimization with Parametric Policies",
    authors: "Xiang Li, Yuheng Zhang, Nan Jiang",
    venue: "arXiv preprint arXiv:2602.23811",
    year: "2026",
    highlight:
      "A unified theory connecting policy-based optimization with value-based offline RL, offering a new perspective on policy-gradient-type methods.",
    links: [{ label: "paper", href: "https://arxiv.org/abs/2602.23811" }],
  },
];
