export type MiscItem = {
  title: string;
  description: string;
  meta?: string;
  href?: string;
};

export const miscellaneous: MiscItem[] = [
  {
    title: "Teaching",
    description:
      "Placeholder for teaching assistantships, guest lectures, tutorials, or mentoring.",
    meta: "Add course names and semesters here.",
  },
  {
    title: "Service",
    description:
      "Placeholder for reviewing, reading groups, student organizations, or conference service.",
    meta: "Keep this concise and dated.",
  },
  {
    title: "Selected Talks",
    description:
      "Placeholder for seminar talks, workshop presentations, or invited lectures.",
    href: "#",
  },
];
