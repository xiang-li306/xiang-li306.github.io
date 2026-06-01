export type BlogPost = {
  title: string;
  date: string;
  summary: string;
  href: string;
  tags?: string[];
};

export const blogPosts: BlogPost[] = [
  {
    title: "Welcome",
    date: "2026-06-01",
    summary:
      "A short placeholder note for future research updates, reading notes, and essays.",
    href: "#",
    tags: ["meta"],
  },
  {
    title: "Reading Notes Placeholder",
    date: "2026-05-15",
    summary:
      "Use this section for informal thoughts on papers, technical ideas, or seminar notes.",
    href: "#",
    tags: ["notes", "research"],
  },
];
