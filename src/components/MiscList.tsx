import type { MiscItem } from "../data/misc";

type MiscListProps = {
  items: MiscItem[];
};

export function MiscList({ items }: MiscListProps) {
  return (
    <div className="grid gap-5 sm:grid-cols-2">
      {items.map((item) => (
        <article key={item.title} className="rounded border border-ink-200 p-5">
          <h3 className="text-base font-semibold text-ink-900">
            {item.href ? (
              <a
                href={item.href}
                className="underline decoration-transparent hover:decoration-ink-700"
              >
                {item.title}
              </a>
            ) : (
              item.title
            )}
          </h3>
          <p className="mt-2 text-sm leading-6 text-ink-700">
            {item.description}
          </p>
          {item.meta ? (
            <p className="mt-3 text-sm leading-6 text-ink-500">
              {item.href ? (
                <a
                  href={item.href}
                  className="underline decoration-ink-200 hover:text-ink-900 hover:decoration-ink-700"
                >
                  {item.meta}
                </a>
              ) : (
                item.meta
              )}
            </p>
          ) : null}
        </article>
      ))}
    </div>
  );
}
