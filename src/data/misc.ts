export type MiscItem = {
  title: string;
  description: string;
  meta?: string;
  href?: string;
};

export const miscellaneous: MiscItem[] = [
  {
    title: "Service",
    description:
      "Reviewer for ICML 2026 and NeurIPS 2026.",
  },
  {
    title: "Bilibili Channel",
    description:
      "I actively maintain a Bilibili channel with Chinese videos on ML theory, RL theory, DL theory, mathematics, and some personal reflections.",
    meta: "Channel link",
    href: "https://space.bilibili.com/32773300?spm_id_from=333.1007.0.0",
  },
];
