import type { BlogPost } from "../data/blog";

type BlogListProps = {
  posts: BlogPost[];
};

export function BlogList({ posts }: BlogListProps) {
  return (
    <div className="divide-y divide-ink-200">
      {posts.map((post) => (
        <article key={post.title} className="py-5 first:pt-0 last:pb-0">
          <div className="flex flex-col gap-1 sm:flex-row sm:items-baseline sm:justify-between">
            <h3 className="text-lg font-semibold leading-7 text-ink-900">
              <a
                href={post.href}
                className="underline decoration-transparent hover:decoration-ink-700"
              >
                {post.title}
              </a>
            </h3>
            <time className="text-sm text-ink-500" dateTime={post.date}>
              {post.date}
            </time>
          </div>
          <p className="mt-2 text-sm leading-6 text-ink-700">{post.summary}</p>
          {post.tags?.length ? (
            <div className="mt-2 flex flex-wrap gap-2">
              {post.tags.map((tag) => (
                <span key={tag} className="text-xs uppercase tracking-wide text-ink-500">
                  {tag}
                </span>
              ))}
            </div>
          ) : null}
        </article>
      ))}
    </div>
  );
}
