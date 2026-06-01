import type { Publication } from "../data/publications";

type PublicationListProps = {
  publications: Publication[];
};

export function PublicationList({ publications }: PublicationListProps) {
  return (
    <ol className="space-y-7">
      {publications.map((publication) => (
        <li key={`${publication.title}-${publication.year}`}>
          <article>
            <h3 className="text-lg font-semibold leading-7 text-ink-900">
              {publication.title}
            </h3>
            <p className="mt-1 text-sm leading-6 text-ink-700">
              {publication.authors}
            </p>
            <p className="mt-1 text-sm leading-6 text-ink-500">
              {publication.venue}, {publication.year}
            </p>
            {publication.note ? (
              <p className="mt-2 text-sm leading-6 text-ink-700">
                {publication.note}
              </p>
            ) : null}
            {publication.links?.length ? (
              <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-sm">
                {publication.links.map((link) => (
                  <a
                    key={link.label}
                    href={link.href}
                    className="text-ink-900 underline decoration-ink-200 hover:decoration-ink-700"
                  >
                    {link.label}
                  </a>
                ))}
              </div>
            ) : null}
          </article>
        </li>
      ))}
    </ol>
  );
}
