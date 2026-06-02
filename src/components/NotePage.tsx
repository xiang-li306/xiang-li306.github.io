import { useEffect } from "react";
import type { BlogPost } from "../data/blog";
import { MarkdownView } from "./MarkdownView";

type NotePageProps = {
  post: BlogPost;
};

declare global {
  interface Window {
    MathJax?: {
      typesetPromise?: () => Promise<void>;
    };
  }
}

export function NotePage({ post }: NotePageProps) {
  useEffect(() => {
    window.MathJax?.typesetPromise?.();
  }, [post.content]);

  return (
    <article className="mx-auto max-w-3xl">
      <a
        href="/#notes"
        className="text-sm font-medium text-ink-700 underline decoration-ink-200 hover:text-ink-900 hover:decoration-ink-700"
      >
        Back to Blog / Notes
      </a>
      <header className="mt-8 border-b border-ink-200 pb-6">
        <h1 className="text-3xl font-semibold tracking-tight text-ink-900 sm:text-4xl">
          {post.title}
        </h1>
        <time className="mt-3 block text-sm text-ink-500" dateTime={post.date}>
          {post.date}
        </time>
      </header>
      <div className="pt-8">
        <MarkdownView content={post.content ?? ""} />
      </div>
    </article>
  );
}
