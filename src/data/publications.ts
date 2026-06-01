export type Publication = {
  title: string;
  authors: string;
  venue: string;
  year: string;
  note?: string;
  links?: {
    label: string;
    href: string;
  }[];
};

export const publications: Publication[] = [
  {
    title: "A Placeholder Paper on Reliable Learning Systems",
    authors: "Xiang Li, Collaborator Name",
    venue: "Preprint",
    year: "2026",
    note: "Replace this entry with a real paper, arXiv preprint, or conference publication.",
    links: [
      { label: "paper", href: "#" },
      { label: "code", href: "#" },
    ],
  },
  {
    title: "Notes on Algorithms, Models, and Generalization",
    authors: "Xiang Li",
    venue: "Working manuscript",
    year: "2025",
    links: [{ label: "draft", href: "#" }],
  },
];
