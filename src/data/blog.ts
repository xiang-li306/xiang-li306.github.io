import sampleMarkdown from "../content/notes/sample-markdown.md?raw";

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
    title: "Sample Markdown Note",
    date: "2026-06-01",
    summary:
      "A sample note rendered from Markdown, including LaTeX math and an image stored with the note assets.",
    slug: "sample-markdown",
    type: "markdown",
    content: sampleMarkdown,
    tags: ["markdown", "latex"],
  },
  {
    title: "Sample PDF Note",
    date: "2026-05-15",
    summary:
      "A sample PDF note. For now, PDF notes download directly; a browser preview can be added later.",
    slug: "sample-pdf",
    type: "pdf",
    href: "/notes/sample-pdf/template-note.pdf",
    tags: ["pdf"],
  },
];
