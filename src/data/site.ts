export type ProfileLink = {
  label: string;
  href: string;
};

export type NavigationLink = {
  label: string;
  href: string;
};

export const siteConfig = {
  navigation: [
    { label: "About", href: "#about" },
    { label: "Publications", href: "#publications" },
    { label: "Blog / Notes", href: "#notes" },
    { label: "Misc.", href: "#misc" },
  ] satisfies NavigationLink[],
  profile: {
    name: "Xiang Li",
    role: "PhD Student in Computer Science",
    affiliation: "Your University / Department",
    location: "City, Country",
    email: "your.email@example.edu",
    photo: "/profile-placeholder.svg",
    researchSummary:
      "I am interested in machine learning, artificial intelligence, and the theoretical foundations of reliable computational systems. My current work explores how to build models and algorithms that are robust, interpretable, and useful in scientific or real-world settings.",
    interests: [
      "Machine learning",
      "Artificial intelligence",
      "Theory and algorithms",
      "Reliable computational systems",
    ],
    links: [
      { label: "Email", href: "mailto:your.email@example.edu" },
      { label: "Google Scholar", href: "https://scholar.google.com/" },
      { label: "GitHub", href: "https://github.com/xiang-li306" },
      { label: "CV", href: "/cv.pdf" },
    ] satisfies ProfileLink[],
  },
};
