import type { ReactNode } from "react";

type SectionProps = {
  id: string;
  title: string;
  description?: string;
  children: ReactNode;
};

export function Section({ id, title, description, children }: SectionProps) {
  return (
    <section id={id} className="border-b border-ink-200 py-11 last:border-b-0">
      <div className="grid gap-7 md:grid-cols-[180px_minmax(0,1fr)]">
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-ink-900">
            {title}
          </h2>
          {description ? (
            <p className="mt-3 text-sm leading-6 text-ink-500">{description}</p>
          ) : null}
        </div>
        <div>{children}</div>
      </div>
    </section>
  );
}
