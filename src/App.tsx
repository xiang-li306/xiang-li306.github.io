import { BlogList } from "./components/BlogList";
import { Footer } from "./components/Footer";
import { Hero } from "./components/Hero";
import { NotePage } from "./components/NotePage";
import { MiscList } from "./components/MiscList";
import { Navbar } from "./components/Navbar";
import { PostsPage } from "./components/PostsPage";
import { PublicationList } from "./components/PublicationList";
import { Section } from "./components/Section";
import { blogPosts } from "./data/blog";
import { miscellaneous } from "./data/misc";
import { publications } from "./data/publications";
import { siteConfig } from "./data/site";
import { useHashRoute } from "./hooks/useHashRoute";

export default function App() {
  const route = useHashRoute();
  const selectedNote = route.startsWith("notes/")
    ? blogPosts.find((post) => post.slug === route.replace("notes/", ""))
    : undefined;
  const isPostsPage = route === "posts";
  const recentPosts = blogPosts.slice(0, 3);

  return (
    <div className="min-h-screen bg-white text-ink-900">
      <Navbar links={siteConfig.navigation} title={siteConfig.profile.displayName} />
      <main className="mx-auto w-full max-w-5xl px-5 pb-16 pt-10 sm:px-8 sm:pt-14">
        {selectedNote?.type === "markdown" ? (
          <NotePage post={selectedNote} />
        ) : isPostsPage ? (
          <PostsPage posts={blogPosts} />
        ) : (
          <>
            <Hero profile={siteConfig.profile} />
            <Section
              id="publications"
              title="Publications"
            >
              <PublicationList publications={publications} />
            </Section>
            <Section
              id="posts-preview"
              title="Posts"
            >
              <BlogList posts={recentPosts} />
              {blogPosts.length > recentPosts.length ? (
                <div className="mt-5">
                  <a
                    href="#posts"
                    className="inline-flex items-center rounded border border-ink-300 px-3 py-2 text-sm font-medium text-ink-800 hover:border-ink-700 hover:text-ink-950"
                  >
                    View all posts
                  </a>
                </div>
              ) : null}
            </Section>
            <Section
              id="misc"
              title="Miscellaneous"
            >
              <MiscList items={miscellaneous} />
            </Section>
          </>
        )}
      </main>
      <Footer profile={siteConfig.profile} />
    </div>
  );
}
