export type ProfileLink = {
  label: string;
  href: string;
};

export type NavigationLink = {
  label: string;
  href: string;
};

export type TextSegment = {
  text: string;
  href?: string;
};

export const siteConfig = {
  navigation: [
    { label: "About", href: "#about" },
    { label: "Publications", href: "#publications" },
    { label: "Posts", href: "#posts" },
    { label: "Misc.", href: "#misc" },
  ] satisfies NavigationLink[],
  profile: {
    name: "Xiang Li",
    displayName: "Xiang Li (\u674e\u60f3)",
    affiliation: {
      name: "Siebel School of Computing and Data Science, UIUC",
      href: "https://siebelschool.illinois.edu/",
    },
    email: "xiang306@illinois.edu",
    photo: "/myphoto.png",
    intro: [
      { text: "Hi, I am Xiang, an incoming PhD student in computer science at UIUC, fortunately advised by " },
      { text: "Prof. Nan Jiang", href: "https://nanjiang.cs.illinois.edu/" },
      { text: ". Previously, I obtained my bachelor's degree from the Kuang Yaming Honors School at Nanjing University, where I worked with " },
      { text: "Prof. Peng Zhao", href: "https://www.pengzhao-ml.com/" },
      { text: "." },
    ] satisfies TextSegment[],
    researchSummary:
      "My research interests lie in machine learning, reinforcement learning, and theory and algorithms. I currently study fundamental reinforcement learning theory, with a particular interest in how these perspectives can help us understand and improve language models. More broadly, I am interested in developing theoretical viewpoints that clarify how AI and machine learning systems behave.",
    interests: [
      "Machine Learning",
      "Reinforcement Learning",
      "Theory and Algorithms",
    ],
    links: [
      { label: "Email", href: "mailto:xiang306@illinois.edu" },
      { label: "Google Scholar", href: "https://scholar.google.com/citations?user=_hxYnNQAAAAJ&hl=en&oi=sra" },
      { label: "GitHub", href: "https://github.com/xiang-li306" },
      { label: "Bilibili", href: "https://space.bilibili.com/32773300?spm_id_from=333.1007.0.0" },
    ] satisfies ProfileLink[],
  },
};
