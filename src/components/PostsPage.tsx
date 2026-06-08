import type { BlogPost } from "../data/blog";
import { BlogList } from "./BlogList";

type PostsPageProps = {
  posts: BlogPost[];
};

export function PostsPage({ posts }: PostsPageProps) {
  return (
    <div className="mx-auto max-w-5xl">
      <header className="border-b border-ink-200 pb-6">
        <a
          href="/"
          className="text-sm font-medium text-ink-700 underline decoration-ink-200 hover:text-ink-900 hover:decoration-ink-700"
        >
          Back to home
        </a>
        <h1 className="mt-8 text-3xl font-semibold tracking-tight text-ink-900 sm:text-4xl">
          Posts
        </h1>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-ink-600">
          Personal blogs and notes, mostly on AI, machine learning, and related topics.
        </p>
      </header>

      <div className="grid gap-10 pt-8 lg:grid-cols-[minmax(0,1fr)_220px]">
        <div>
          <BlogList posts={posts} />
        </div>
        <aside className="lg:sticky lg:top-24 lg:self-start">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-ink-500">
            All posts
          </h2>
          <nav className="mt-4 border-l border-ink-200 pl-4">
            <ol className="space-y-3">
              {posts.map((post) => {
                const href =
                  post.type === "markdown" ? `#/notes/${post.slug}` : post.href ?? "#";

                return (
                  <li key={post.slug}>
                    <a
                      href={href}
                      download={post.type === "pdf" ? true : undefined}
                      className="block text-sm leading-5 text-ink-700 hover:text-ink-900 hover:underline"
                    >
                      {post.title}
                    </a>
                    <time className="mt-1 block text-xs text-ink-500" dateTime={post.date}>
                      {post.date}
                    </time>
                  </li>
                );
              })}
            </ol>
          </nav>
        </aside>
      </div>
    </div>
  );
}
